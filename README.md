# Transformer

---

This repo contains the Transformer architecture as described in [Attention is All You Need](https://arxiv.org/abs/1706.03762). All of the code to run the model is housed in _transformer.py_.

#### Requirements

---

This build assumes Python >= 3. We also want to use the TensorFlow Beta 2.0.0 (2019). Check the [TensorFlow Install Instructions](https://www.tensorflow.org/install/pip) for up to date instructions.

```
# For CPU

pip install --upgrade pip
pip install tensorflow==2.0.0-rc0 tensorflow-datasets

# For GPU

pip install --upgrade pip
pip install tensorflow-gpu==2.0.0-rc0 tensorflow-datasets
```

If you have _Docker_ installed, use the provided docker image to run your model! The following will build and then run the image. You do not need to rebuild the container every time.

```
# Build and Run

sh build && sh run
```

To re-run just use:

```
sh run
```

#### Run

---

To use this model create a config JSON file like this:

```
{
  "seed": 420365,
  "checkpoint": false,
  "layers": 1,
  "attn-heads": 16,
  "model-depth": 32,
  "dff": 64,
  "dropout": 0.1,
  "epochs": 300,
  "batch-size": 1
}

// output to config.json
```

From this point, you can run the following command. This will use your config.json by default. Make sure you set up your data pipeline in transformer.py beforehand.

```
python transformer.py
```
