# cifarium

Find the smallest AI model that handles your examples right in the browser, and get a folder you can drop into your site.

> **Early development.** Today only `cifarium init` and the temporary `cifarium smoke` work. The first usable version will be 0.1.0; until then the file formats may still change.

## Why

A small model can run in the visitor's browser instead of on a server: no cost per request, and the text never leaves the device. Today you have to pick the model yourself, prepare it, and hope it works for your visitors. cifarium does that part, starting from a dozen of your own examples.

## Who it is for

Web developers who want a short, frequent AI feature without a server and without knowing which model to pick: sorting support requests by topic first, search by meaning and short summaries later.

## How it will work

```
cifarium init                  # write cifarium.yaml and examples.jsonl to start from
cifarium build                 # pick the smallest model that passes your examples
cifarium check ./cifarium-out  # run the built folder in a real browser
```

`cifarium.yaml`:

```yaml
task: classify               # search and summarize come later
examples: ./examples.jsonl   # 10 to 20 of your own examples
labels:
  billing: payments, invoices and refunds
  bug: something is broken or does not work
pass_rate: 0.85              # share of the examples the model must get right
limits:
  max_size_mb: 300           # first download for a visitor, engine included
output: ./cifarium-out
```

One line of `examples.jsonl`:

```json
{"input": "I was charged twice this month.", "expected": "billing"}
```

`build` tries models from a catalog, smallest first, in your installed Chrome and keeps the first one that passes. It writes a folder with the model, the engine ([Transformers.js](https://github.com/huggingface/transformers.js)), a small runtime, an example page and a report. Your site serves it as static files.

## Try it from source

Needs Python 3.10+ and an installed Google Chrome.

```
git clone https://github.com/Cifarium/cifarium
cd cifarium
python -m pip install -e .
cifarium init
```

## License

Apache-2.0. Models keep their own licenses; by default cifarium only picks models whose license allows commercial use.
