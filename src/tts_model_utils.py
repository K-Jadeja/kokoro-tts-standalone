import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
from loguru import logger


def download_file(url: str, output_path: Path, show_progress: bool = True) -> bool:
    """
    Download a file from URL with progress bar.

    Args:
        url: Download URL
        output_path: Where to save the file
        show_progress: Whether to show progress bar

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if show_progress:
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"📥 Downloading {output_path.name}",
            )

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    if show_progress:
                        progress_bar.update(len(chunk))

        if show_progress:
            progress_bar.close()

        logger.success(f"✅ Downloaded: {output_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {str(e)}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract tar.bz2, tar.gz, or zip archive.

    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.name.endswith(".tar.bz2"):
            with tarfile.open(archive_path, "r:bz2") as tar:
                tar.extractall(extract_to)
        elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(
            ".tgz"
        ):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(extract_to)
        else:
            logger.error(f"❌ Unsupported archive format: {archive_path}")
            return False

        logger.success(f"✅ Extracted: {archive_path} to {extract_to}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to extract {archive_path}: {str(e)}")
        return False


def download_and_extract_model(
    url: str,
    models_dir: str | Path = "models",
    model_name: str = None,
    keep_archive: bool = False,
) -> Path | None:
    """
    Download and extract a TTS model archive.

    Args:
        url: Download URL for the model archive
        models_dir: Directory to store models
        model_name: Expected model directory name (if different from archive name)
        keep_archive: Whether to keep the downloaded archive file

    Returns:
        Path to extracted model directory or None if failed
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # Determine archive filename and model directory name
    archive_filename = url.split("/")[-1]
    archive_path = models_path / archive_filename

    if model_name:
        expected_model_dir = models_path / model_name
    else:
        # Remove archive extensions to get model directory name
        base_name = archive_filename
        for ext in [".tar.bz2", ".tar.gz", ".tgz", ".zip"]:
            if base_name.endswith(ext):
                base_name = base_name[: -len(ext)]
                break
        expected_model_dir = models_path / base_name

    # Check if model already exists
    if expected_model_dir.exists() and any(expected_model_dir.iterdir()):
        logger.info(f"✅ Model already exists: {expected_model_dir}")
        return expected_model_dir

    # Download the archive
    logger.info(f"📥 Downloading TTS model from: {url}")
    if not download_file(url, archive_path):
        return None

    # Extract the archive
    logger.info("📦 Extracting model archive...")
    if not extract_archive(archive_path, models_path):
        return None

    # Clean up archive if requested
    if not keep_archive and archive_path.exists():
        archive_path.unlink()
        logger.info(f"🗑️ Removed archive: {archive_path}")

    # Verify extraction
    if expected_model_dir.exists() and any(expected_model_dir.iterdir()):
        logger.success(f"🎉 Model ready: {expected_model_dir}")
        return expected_model_dir
    else:
        logger.error(
            f"❌ Model extraction failed or directory not found: {expected_model_dir}"
        )
        return None


def verify_kokoro_model(model_path: Path) -> bool:
    """
    Verify that a Kokoro model directory contains required files.

    Args:
        model_path: Path to model directory

    Returns:
        bool: True if all required files exist
    """
    required_files = ["model.onnx", "tokens.txt", "voices.bin", "espeak-ng-data"]

    missing_files = []
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"❌ Missing required files in {model_path}: {missing_files}")
        return False

    logger.success(f"✅ Kokoro model verified: {model_path}")
    return True


# Predefined model configurations
KOKORO_MODELS = {
    "kokoro-multi-lang-v1_0": {
        "url": "https://github.com/K-Jadeja/tts-models/releases/download/v1.1/kokoro-multi-lang-v1_0.tar.bz2",
        "description": "Multilingual Kokoro TTS model v1.0",
        "size_mb": 333.2,
        "sha256": "3ca17094afa7fd40c51d5ed78fd938087ead13c2ac6a3cc2612cc41e4d538bb7",
    },
    "kokoro-en-v0_19": {
        "url": "https://github.com/your-username/kokoro-models/releases/download/v1.0/kokoro-en-v0_19.tar.bz2",
        "description": "English Kokoro TTS model v0.19",
        "size_mb": 609.6,
        "sha256": "f021619e18131a9bac5029166c89f397818693e25f05c73c24a34ad077d57bc4",
    },
    "kokoro-int8-multi-lang-v1_0": {
        "url": "https://github.com/your-username/kokoro-models/releases/download/v1.0/kokoro-int8-multi-lang-v1_0.tar.bz2",
        "description": "INT8 quantized multilingual Kokoro TTS model v1.0",
    },
    "kokoro-int8-multi-lang-v1_1": {
        "url": "https://github.com/your-username/kokoro-models/releases/download/v1.1/kokoro-int8-multi-lang-v1_1.tar.bz2",
        "description": "INT8 quantized multilingual Kokoro TTS model v1.1",
    },
}


def download_kokoro_model(
    model_name: str, models_dir: str | Path = "models"
) -> Path | None:
    """
    Download a predefined Kokoro model.

    Args:
        model_name: Name of the model to download
        models_dir: Directory to store models

    Returns:
        Path to downloaded model directory or None if failed
    """
    if model_name not in KOKORO_MODELS:
        logger.error(f"❌ Unknown Kokoro model: {model_name}")
        logger.info(f"Available models: {list(KOKORO_MODELS.keys())}")
        return None

    model_config = KOKORO_MODELS[model_name]
    logger.info(f"📦 Downloading {model_config['description']}")

    model_path = download_and_extract_model(
        url=model_config["url"], models_dir=models_dir, model_name=model_name
    )

    if model_path and verify_kokoro_model(model_path):
        return model_path
    else:
        logger.error(f"❌ Failed to download or verify {model_name}")
        return None


if __name__ == "__main__":
    # Test downloading a model
    result = download_kokoro_model("kokoro-multi-lang-v1_0", "./models")
    if result:
        print(f"✅ Model downloaded to: {result}")
    else:
        print("❌ Download failed")
