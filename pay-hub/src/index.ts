import { Hono } from "hono";
import { cors } from "hono/cors";
import type { AppEnv } from "./types.js";
import checkout from "./routes/checkout.js";
import order from "./routes/order.js";
import webhook from "./routes/webhook.js";

const app = new Hono<AppEnv>();

// CORS for landing page
app.use(
  "*",
  cors({
    origin: [
      "https://neuralmemory.theio.vn",
      "https://companion.theio.vn",
      "https://nhadaututtheky.github.io",
      "http://localhost:3000",
    ],
    allowMethods: ["GET", "POST", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
  }),
);

// Health
app.get("/", (c) =>
  c.json({
    name: "Neural Memory Pay Hub",
    version: "1.0.0",
    status: "healthy",
  }),
);

app.get("/health", (c) => c.json({ status: "ok" }));

// Routes
app.route("/checkout", checkout);
app.route("/order", order);
app.route("/webhook", webhook);

// Pro features granted on all verified licenses
const PRO_FEATURES = [
  "merkle_sync",
  "cone_queries",
  "directional_compression",
  "cross_encoder",
  "smart_merge",
  "infinity_db",
];

// ── Shared verify logic ────────────────────────────────────────────────────

async function verifyKey(c: any, key: string) {
  if (!key) {
    return c.json({ valid: false, error: "Missing key" });
  }

  const normalizedKey = key.startsWith("nm_")
    ? key.replaceAll("_", "-").toUpperCase()
    : key.toUpperCase();

  // Companion keys (CPN-*) → forward to companion-verify via Service Binding
  if (normalizedKey.startsWith("CPN-")) {
    try {
      const worker = c.env.COMPANION_WORKER;
      const res = await worker.fetch(
        new Request(`https://companion.theio.vn/verify?key=${encodeURIComponent(key)}`),
      );
      const data = await res.json();
      return c.json(data, res.status);
    } catch {
      return c.json({ valid: false, error: "Companion verify unreachable" }, 502);
    }
  }

  // NM keys → D1 primary, XLabs fallback
  const db = c.env.PAY_DB;
  const d1Order = await db
    .prepare(
      "SELECT product, license_key, fulfilled_at FROM orders WHERE UPPER(license_key) = ? AND status = 'fulfilled'",
    )
    .bind(normalizedKey)
    .first<{ product: string; license_key: string; fulfilled_at: string }>();

  if (d1Order) {
    const tier = d1Order.product.includes("TEAM") ? "team" : "pro";
    return c.json({
      valid: true,
      tier,
      expires_at: null,
      features: PRO_FEATURES,
    });
  }

  try {
    const res = await fetch("https://admin.theio.vn/api/licenses", {
      headers: { Authorization: `Bearer ${c.env.XLABS_API_KEY}` },
    });

    if (res.ok) {
      const data = await res.json<{
        data: Array<{
          license_key: string;
          project_slug: string;
          tier: string;
          status: string;
          features_json: string;
          expires_at: string | null;
        }>;
      }>();

      const match = data.data?.find(
        (l) =>
          l.license_key.toUpperCase() === normalizedKey &&
          l.project_slug === "neural-memory" &&
          l.status === "active",
      );

      if (match) {
        let features: string[] = [];
        try {
          features = JSON.parse(match.features_json || "[]");
        } catch {
          features = PRO_FEATURES;
        }
        return c.json({
          valid: true,
          tier: match.tier,
          expires_at: match.expires_at,
          features,
        });
      }
    }
  } catch {
    // XLabs API unreachable
  }

  return c.json({ valid: false, error: "Invalid or expired license key" });
}

// ── License sync relay (XLabs → companion-verify via Service Binding) ──────

app.post("/admin/license/sync", async (c) => {
  // Auth: XLabs webhook secret or companion admin secret
  const auth = c.req.header("Authorization");
  const expected = `Bearer ${c.env.COMPANION_ADMIN_SECRET}`;
  if (auth !== expected) {
    return c.json({ error: "Unauthorized" }, 401);
  }

  const body = await c.req.json<{
    action: "create" | "revoke";
    license_key: string;
    tier?: string;
    email?: string;
    name?: string;
    max_sessions?: number;
    expires_at?: string;
    duration_days?: number;
  }>();

  const worker = c.env.COMPANION_WORKER;

  if (body.action === "create") {
    let durationDays = body.duration_days ?? 370;
    if (body.expires_at) {
      const ms = new Date(body.expires_at).getTime() - Date.now();
      durationDays = Math.max(1, Math.ceil(ms / 86_400_000));
    }

    const res = await worker.fetch(
      new Request("https://companion.theio.vn/admin/create", {
        method: "POST",
        headers: {
          Authorization: expected,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          key: body.license_key,
          tier: body.tier ?? "pro",
          email: body.email ?? "",
          durationDays,
        }),
      }),
    );
    const data = await res.json();
    return c.json(data, res.status);
  }

  if (body.action === "revoke") {
    const res = await worker.fetch(
      new Request("https://companion.theio.vn/admin/revoke", {
        method: "POST",
        headers: {
          Authorization: expected,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ key: body.license_key }),
      }),
    );
    const data = await res.json();
    return c.json(data, res.status);
  }

  return c.json({ error: "Invalid action" }, 400);
});

// Verify endpoint — supports both GET (Companion app) and POST (legacy)
app.get("/verify", async (c) => {
  const key = (c.req.query("key") || "").trim();
  return verifyKey(c, key);
});

app.post("/verify", async (c) => {
  const body = await c.req.json<{ key: string }>();
  const key = (body.key || "").trim();
  return verifyKey(c, key);
});

export default app;
