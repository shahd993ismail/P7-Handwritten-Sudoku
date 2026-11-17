"""NN tests: model shape, probabilities, blank rule, image reading, accuracy."""
import numpy as np
import pytest
import torch
from PIL import Image

import config
from data.puzzle_gen import IMAGE_PX, render_mosaic
from models.cnn_model import DigitCNN
from nn.predict import cell_probabilities, is_blank, load_model, predict_proba

needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="run python -m nn.train first")
needs_data = pytest.mark.skipif(not config.PROCESSED_FILE.exists(), reason="run python -m data.make_data first")


def test_cnn_output_shape_and_softmax():
    m = DigitCNN()
    x = torch.rand(5, 1, 28, 28)
    assert m(x).shape == (5, 9)
    p = m.predict_proba(x)
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-5) and (p >= 0).all()


def test_blank_rule():
    assert is_blank(np.zeros((28, 28)))
    img = np.zeros((28, 28))
    img[6:22, 13:15] = 1.0          # a thin vertical stroke, like a "1"
    assert not is_blank(img)


@needs_model
def test_cell_probabilities_blank_rows_are_zero():
    cells = np.zeros((81, 28, 28), np.float32)
    cells[0, 6:22, 13:15] = 1.0
    out = cell_probabilities(cells, load_model())
    assert out["blank"][0] is False and all(out["blank"][1:])
    assert np.isclose(out["probs"][0].sum(), 1.0) and out["probs"][1:].sum() == 0
    assert out["grid"][1:] == [0] * 80


@needs_model
@needs_data
def test_cnn_accuracy_on_test_subset():
    from data.make_data import load_processed
    d = load_processed()
    probs = predict_proba(load_model(), d["x_test"][:2000])
    acc = (probs.argmax(1) + 1 == d["y_test"][:2000]).mean()
    assert acc > 0.97


@needs_model
@needs_data
def test_image_reading_generated_and_resized():
    from data.make_data import load_processed
    from data.puzzle_gen import generate_puzzle, sample_digit_images
    from integration.pipeline import recognise_image
    d = load_processed()
    rng = np.random.default_rng(11)
    puzzle, _ = generate_puzzle(32, rng)
    imgs, _ = sample_digit_images(puzzle, d["x_test"], d["y_test"], rng)
    mosaic = render_mosaic(imgs)
    assert mosaic.shape == (IMAGE_PX, IMAGE_PX)
    rec = recognise_image(Image.fromarray(mosaic))
    assert rec["extraction"] == "exact generated layout"
    assert sum(a != b for a, b in zip(rec["grid"], puzzle)) <= 1
    # a resized copy on a larger white sheet uses grid detection
    sheet = Image.new("L", (520, 520), 255)
    sheet.paste(Image.fromarray(mosaic).resize((450, 450)), (35, 35))
    rec2 = recognise_image(sheet)
    assert rec2["extraction"].startswith("grid detection")
    assert [v > 0 for v in rec2["grid"]] == [v > 0 for v in puzzle]   # blanks found exactly
    assert sum(a != b for a, b in zip(rec2["grid"], puzzle)) <= 3


def test_bad_image_rejected():
    from integration.pipeline import extract_cells
    with pytest.raises(ValueError):
        extract_cells(np.zeros((10, 10)))
