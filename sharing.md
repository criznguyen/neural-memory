
**Claude quên sạch mọi thứ sau mỗi session — đây là open-source giúp nó "có não" thật sự 🧠**

Dùng Claude Code / Cursor / Windsurf một thời gian, mình nhận ra một nỗi đau chung: **AI rất giỏi, nhưng nó mất trí nhớ mỗi lần mở session mới**. Bạn đã quyết định dùng Postgres thay SQLite? Quên. Bạn đã fix con bug auth tuần trước? Quên. Bạn đã dặn nó đừng viết comment thừa? Quên luôn.

Cách sửa phổ thông: nhét vector DB + RAG + embedding API. Nghe hay, nhưng:
- Mỗi query tốn $$$ cho OpenAI embedding
- Chỉ match "gần giống" theo similarity, không hiểu **quan hệ nhân quả**
- Multi-hop reasoning → phải query nhiều lần, vẫn sai
- Data của bạn chạy qua server bên thứ ba

Mấy tháng nay mình xài một project open-source Việt Nam tên là **Neural Memory** (github.com/nhadaututtheky/neural-memory) — và nó giải bài toán theo cách khác hoàn toàn.

---

## Ý tưởng: bắt chước não người, không phải search engine

Não bạn không "SELECT WHERE content LIKE '%Alice%'". Bạn nghĩ đến Alice → tự động kích hoạt khuôn mặt, cuộc nói chuyện cuối, dự án làm chung. Đó là **spreading activation** — tín hiệu lan qua mạng neuron.

Neural Memory làm đúng như vậy:
- **Neurons** — atomic units: người, thời gian, khái niệm, hành động, trạng thái
- **Synapses** — 24 loại quan hệ có gán nhãn: `CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS`...
- **Fibers** — đường đi tín hiệu, dùng nhiều → "dẫn điện" tốt hơn (Hebbian learning đúng nghĩa)

Khi bạn hỏi *"Tại sao outage thứ 3 xảy ra?"*, vector DB trả về câu gần giống nhất. Neural Memory trace ra chuỗi:

```
outage ← CAUSED_BY ← JWT expiry ← SUGGESTED_BY ← review của Alice
```

Nó không **tìm**, nó **suy luận**.

---

## Con số khiến mình chuyển qua

Benchmark head-to-head với Mem0 (memory tool phổ biến nhất hiện nay):

| | Neural Memory | Mem0 |
|--|--|--|
| Write 50 memories | **1.22s** | 148.16s (**121x chậm hơn**) |
| Multi-hop reasoning | **0.417** | 0.383 |
| API calls | **0** | 70 |
| Cost/10K ops/day | **$0.00** | ~$2–5/day |

Mem0 mỗi lần `add()` gọi 1 LLM call để extract memory → đắt + chậm. Neural Memory encode trực tiếp vào graph → 0 đồng, chạy offline hoàn toàn.

---

## Điều làm mình ấn tượng nhất

**1. Chỉ cần 3 tools.** Có 60 MCP tools cả thảy, nhưng 95% thời gian bạn chỉ cần:
- `nmem_remember` — lưu, auto-detect type
- `nmem_recall` — truy hồi, spreading activation
- `nmem_health` — điểm sức khỏe brain A-F, có gợi ý fix

Còn lại — consolidation, decay, reinforcement — chạy nền tự động.

**2. Memory có vòng đời thật.** Ebbinghaus decay curve cho ký ức không dùng. Hebbian reinforcement cho ký ức hay dùng. Consolidation gom ký ức episodic cũ thành semantic knowledge. Contradiction detection auto-resolve khi bạn thay đổi quyết định. Đây là thứ RAG không bao giờ làm được.

**3. Self-host sync qua Cloudflare của CHÍNH BẠN.** Không phải SaaS. Bạn deploy sync hub lên Cloudflare Worker free tier, D1 của bạn, encryption key của bạn. Merkle delta chỉ đồng bộ phần khác biệt → nhanh, riêng tư tuyệt đối. Laptop ↔ Desktop ↔ phone — không qua server bên thứ ba nào cả.

**4. Train từ docs/code/schema thật.** Feed PDF/DOCX/PPTX/HTML → biến thành knowledge permanent. Train từ Postgres schema → agent hiểu FK relationships, audit trails, soft-delete patterns. Đây là cách làm "internal knowledge base" thật sự cho team.

**5. Cognitive reasoning.** Không chỉ store-retrieve. Nó có `nmem_hypothesize`, `nmem_evidence`, `nmem_predict`, `nmem_verify` với Bayesian confidence. Bạn để AI tự hình thành giả thuyết, nộp evidence, và verify.

---

## Setup 30 giây với Claude Code

```bash
/plugin marketplace add nhadaututtheky/neural-memory
/plugin install neural-memory@neural-memory-marketplace
```

Với Cursor/Windsurf:
```bash
pip install neural-memory
```

Thêm vào config:
```json
{
  "mcpServers": {
    "neural-memory": { "command": "nmem-mcp" }
  }
}
```

Khởi động lại. Xong. Không cần `init`, MCP server auto-initialize lần đầu gọi.

---

## Mình khuyên dùng khi nào?

- ✅ Dev solo hoặc team nhỏ, muốn AI "nhớ" context dự án dài hạn
- ✅ Không muốn tốn tiền embedding API
- ✅ Cần data ở lại máy mình (data sovereignty, client nhạy cảm)
- ✅ Cần multi-hop reasoning (bug root cause, decision history)
- ✅ Muốn multi-device sync mà không qua cloud SaaS

## Khi nào cân nhắc?

- ⚠️ Brain của bạn > 100K memory và cần semantic search nghĩa chính xác → cần bản Pro ($9/mo, kích hoạt InfinityDB + HNSW). Nhưng bản Free hoàn toàn đủ dùng cho đa số use case.

---

## Tóm lại

**Neural Memory = MIT license, offline, $0/month, graph-based reasoning, self-sovereign.**

Nếu bạn đang dùng Claude Code và cảm thấy bực vì phải giải thích lại context mỗi session — thử xem. Trải nghiệm "AI nhớ mình nói gì tuần trước" nó khác biệt hơn bạn tưởng rất nhiều.

👉 Repo: https://github.com/nhadaututtheky/neural-memory
👉 Docs: https://nhadaututtheky.github.io/neural-memory/

Nếu thấy hay, cho họ 1 star để project sống lâu 🌟 — đây là một trong những MCP server chất lượng nhất mình từng dùng, và của người Việt làm.

#AI #Claude #ClaudeCode #MCP #OpenSource #NeuralMemory #DeveloperTools
