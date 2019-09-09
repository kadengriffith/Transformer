# Transformer

---

This repo contains the Transformer architecture as described in [Attention is All You Need](https://arxiv.org/abs/1706.03762). All of the code to run the model is housed in _transformer.py_. Using a config.json will make everything easily adjustable for your task.

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

#### Run

---

From this point, you can run the following command in the container.

```
# This will use your config.json by default.
# Make sure you set up your data beforehand.
python transformer.py
```
