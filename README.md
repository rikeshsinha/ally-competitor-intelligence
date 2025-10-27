# Ally Competitor Content Intelligence 

A demo skill made with streamlit UI for comparing a client SKU vs. a competitor’s SKU, checking Amazon style rules, drafting LLM-powered edits, and producing a final Markdown summary + email.

## What it does
- Load a CSV of SKUs that includes both client and competitor products.
- Let you pick a client SKU and a competitor SKU for a head-to-head comparison.
- Run rule checks based on the provided Amazon Pet Supplies style guide.
- Compare PDP fields (title, bullets, description, images).
- Ask an LLM to draft brand- and Amazon-compliant edits for the client SKU (title/bullets/description).
- Pause for human approval before finalizing changes.
- Generate a final Markdown summary and optional email draft for download once you approve the edits.

## Run locally
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # or set in the app sidebar and validate
streamlit run app.py
```

## Expected inputs
Place the following assets in the same folder as `app.py`:
- `asin_data_filled.csv`
- `PetSupplies_PetFood_Styleguide_EN_AE._CB1198675309_.pdf` (for reference only—the rules are encoded in the app.)

## Notes
- If no API key is set, the app will generate heuristic (non-LLM) suggestions so you can still demo the flow.
- CSV column names are auto-detected; expected columns include: `sku_id`, `title`, `bullets`, `description`, `image_urls`, `brand`, `category`.
- Bullets can be delimited by newlines, `|`, `•`, or semicolons; the app attempts to auto-split them.
