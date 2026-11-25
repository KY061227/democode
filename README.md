# GitHub Pages with OPT predictions

This version runs the group's fine-tuned OPT model in a Web Worker using ONNX Runtime Web. GitHub Pages serves the interface and converted model. A Python backend is not used by this interface. Predictions run on each visitor's device after they load the website and click Load model. The adapter is unchanged, but ONNX quantization changes numerical behaviour and timing, so previous Python benchmark results must not be presented as browser measurements.

## Upload and deploy

1. Open the KY061227/democode repository on GitHub.
2. Replace .github/workflows/deploy-ui.yml with the supplied workflow.
3. Add scripts/export_browser.py. Replace web/index.html, web/app.js, web/app.css, and add web/model-worker.js. The bundle is also included; Actions rebuilds it.
4. Keep models/opt-125m-lora-adapter and all its existing files at the repository root. The export script reads this directory.
5. Disable any older Pages workflow in Actions to avoid competing deployments.
6. In Settings > Pages, choose GitHub Actions as the source.
7. Commit the files to main. Open Actions > Build OPT model and publish website.
8. Wait for the model export, worker bundle, artifact upload and deployment jobs to finish. The export downloads OPT, merges the supplied adapter, creates the quantized ONNX file, and validates variable-length inputs before publishing.
9. Open Settings > Pages > Visit site. Refresh the website to get the new JavaScript.
10. Click Load model in the Prediction engine panel. Wait for the download and initialization status to show Model ready. Try a prompt or type Do you followed by a space.

Upload the contents of this project folder directly into the repository root. The workflow assumes web/, scripts/, and models/ are at the root. Keep the dot-prefixed .github directory when uploading. If the GitHub upload picker hides it, edit .github/workflows/deploy-ui.yml directly through GitHub's editor.

Do not upload web/model/model.onnx into GitHub through the file picker. Actions generates that large file and publishes it as a Pages artifact. The generated model is about 272 MiB. Initial loading can take minutes on slower connections. Model download progress is shown. The website remains online without running a server on your laptop. Each browser needs enough memory to load and execute OPT. Desktop Chrome/Edge is the intended demonstration environment.

## Preserved features

Next-word and prefix suggestions, capped user frequency boosting, confirmed vocabulary, optional spelling corrections above Suggestions, punctuation, separate two-person drafts and caches, and sent messages below Suggestions are preserved. Spelling follows the supplied edit-distance-1 dictionary candidates and contextual OPT scoring. Candidate words are scored sequentially in the worker rather than in a Python batch. This can increase latency. The browser engine uses a bounded multi-token shortlist.

The model selector is fixed to the converted LoRA model. Generating several full model downloads would increase site size. The actual browser model is labelled ONNX INT8. Spelling remains a completed-word check triggered after whitespace. Real-word errors such as form/from are a limitation. The battery example may still rank problem first; this migration addresses hosting and model loading rather than fine-tuning quality.

The Python files remain available for the earlier local-server version, but the deployed browser interface uses the worker. The original Streamlit app.py is not needed for Pages.

## Checks performed

The group's adapter was merged and exported locally. The model passed ONNX Runtime variable-length smoke tests. The worker ran real ONNX WebAssembly inference in Node.js, including Do you, Waht, Wht and multiple misspellings. Full interactive browser and live GitHub deployment verification remain pending. Browser speed depends on the visitor's hardware and is not comparable to the earlier CPU Python figures.
