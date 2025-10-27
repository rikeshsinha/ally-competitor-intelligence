# Content Quality Judge (G-Eval style)

Use this rubric if you enable the optional LLM-as-a-judge second pass.

Return JSON: `{"scores":{"title":int,"bullets":int,"desc":int}, "notes": string}`

- **Title (0–5):** key attributes (brand, format, size/pack) present & concise; avoids promo; <= rule cap.
- **Bullets (0–5):** 3–5 items; start with capital; clear benefits; no end punctuation; no promo/seller.
- **Description (0–5):** plain, specific; follows cap; avoids promo/seller; no ALL CAPS.

Explain any deductions in `notes`.
