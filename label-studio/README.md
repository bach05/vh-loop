# Label Studio + SAM2 Setup

## Overview

This setup uses two separate Conda environments:

- `ls-ui` for Label Studio
- `ls-sam2` for the SAM2 backend

The environments are not compatible with each other.

---

## 1) Install Label Studio

Create and activate the UI environment.

```bash
conda create -n ls-ui python=3.11
conda activate ls-ui
pip install label-studio
label-studio start
```

Label Studio opens automatically at:

```text
http://localhost:8080
```

Create a local account with any email and password.

To create the access token:

1. Open the user icon in the top-right corner.
2. Go to **Account & Settings**.
3. Open **Personal Access Token**.
4. Create and copy the token.

The token is in **JWT** format.

A legacy token is still available at:

```text
http://localhost:8080/api/current-user/token
```

Legacy tokens are no longer supported.

---

## 2) Install the SAM2 backend

Create and activate the backend environment.

```bash
conda create -n ls-sam2 python=3.11
conda activate ls-sam2
```

Install PyTorch.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Move to the working directory.

```bash
cd C:/ITR/
```

Clone the official SAM2 repository and install it.

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

Return to the working directory.

```bash
cd C:/ITR/
```

Clone the Label Studio ML backend and install it.

```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend
pip install -e .
```

Create the checkpoint folder.

```bash
cd C:/ITR/label-studio-ml-backend/label_studio_ml/examples
mkdir checkpoints
cd checkpoints
```

Download the SAM2 weights to:

```text
C:\ITR\label-studio-ml-backend\label_studio_ml\examples\checkpoints\sam2.1_hiera_large.pt
```

```powershell
powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt' -OutFile 'sam2.1_hiera_large.pt'"
```

Start the backend.

```bash
cd ..
label-studio-ml start ./segment_anything_2_image -p 9090
```

For CPU-only systems, add:

```bash
--extra-params '{"device": "cpu"}'
```

---

## 3) Connect the backend in Label Studio

After the backend starts on port `9090`, open Label Studio at `http://localhost:8080`.

Create a project, then go to:

**Settings → Model → Connect Model**

Enter:

```text
http://localhost:9090
```

---

## 4) Health checks

Check the backend process:

```bash
label-studio-ml start .
```

Test the backend directly:

```text
http://localhost:9090/health
```
