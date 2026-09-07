AI Engine Air Filter POC

Files required in the repository root:
- index.html
- app.js
- best.onnx

The browser app first uses a zero-shot CLIP gate to check whether the uploaded image appears to be a car engine air filter. If accepted, it runs the user's ONNX classification model and returns GOOD or BAD with confidence.

Proof of concept only. Results are not a substitute for a physical service inspection.
