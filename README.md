---
title: Style Strand AI
emoji: "✂️"
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Style Strand AI

Style Strand AI provides photo-based hair texture, color, and haircut suggestions, with a separate image-model condition screen.

## Photo privacy and limitations

Uploaded photos are sent to this Space for processing. The app does not intentionally save uploaded photos to persistent storage, but images may remain in active session memory while the app is running. Avoid uploading photos unless you are comfortable with this processing.

The condition screen is an AI image-model estimate, not a medical diagnosis. It cannot confirm or rule out a condition; consult a qualified healthcare professional about concerns.

## Runtime

The Docker image uses Python 3.10 and the pinned packages in `requirements.txt`. It includes the inference models required by the app and selected haircut reference images from `app/assets/haircut_references`; training datasets and training outputs are not included.

To run locally with Docker:

```sh
docker build -t style-strand-ai .
docker run --rm -p 7860:7860 style-strand-ai
```
