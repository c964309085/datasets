# Performances tips

TFDS is a wrapper around `tf.data.Dataset`. As such, all performance advices
from the
[tf.data guide](https://www.tensorflow.org/guide/data_performance#optimize_performance)
are still valid.

This document describe the TFDS specific performance tips.

## Small datasets (< GB)

All TFDS dataset generate the data as
[TFRecord files](https://www.tensorflow.org/tutorials/load_data/tfrecord). For
small datasets (e.g. Mnist, Cifar,...), reading from `.tfrecord` can add a
significant overhead.

As those datasets fits in memory, it is possible to significantly improving the
performance by caching or pre-loading the dataset. Note that TFDS automatically
cache small datasets.

### Caching the dataset

Here is a full pipeline example which explicitly cache the dataset after
normalizing the images.

```python
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


ds, ds_info = tfds.load(
    'mnist',
    split='train',
    as_supervised=True,  # returns `(img, label)` instead of dict(image=, ...)
    with_info=True,
)
# Applying normalization before `ds.cache()` to re-use it.
# Warning: Random transformations (e.g. images augmentations) should be applied
# after both `ds.cache()` and `ds.batch()`.
ds = ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = ds.cache()
# For true randomness, we set the shuffle buffer to the full dataset size.
# To get the number of examples, `tf.data.experimental.cardinality(ds)` could
# be used too.
ds = ds.shuffle(ds_info.splits['train'].num_examples)
# Batch after shuffling to get unique batches at each epoch.
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
```

When iterating over this dataset, the second iteration should be much faster
than the first one thanks to the caching.

### Auto-caching

By default, TFDS tries to auto-cache datasets which satisfy the following
constraint:

*   Total dataset size (all splits) < 250 MiB
*   `shuffle_files` disabled OR only a single shard is read

It is possible to opt-out from auto-caching by passing
`read_config=tfds.ReadConfig(try_autocaching=False)` to `tfds.load`. Have a look
at the catalog documentation to see if a specific dataset will use auto-cache.

### Loading the full data as tensor

To avoid the overhead of `tf.data.Dataset` pipeline, you may want to load the
full data as `np.array`. It is possible by setting `batch_size=-1` to batch all
examples in a single `tf.Tensor`. Then use `tfds.as_numpy` for the convertion
`tf.Tensor` -> `np.array`.

```
(img_train, label_train), (img_test, label_test) = tfds.as_numpy(tfds.load(
    'mnist',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))
```

## Large datasets

Large datasets are sharded (split in multiple files), and typically do not fit
in memory so cannot be cached.

### Shuffle and training

Large dataset are sharded in multiple files. During training, it is important to
set the `shuffle_files=True` argument to True. Otherwise epochs will read the
shards in the same orders, and data won't be trully randomized. This can
negativelly impact the loss. Using `ds.shuffle` is not enough as the shuffle
buffer size cannot fit the full dataset.

```
ds = tfds.load('imagenet2012', split='train', shuffle_files=True)
```

Additionally, when `shuffle_files=True`, TFDS disable
[option.experimental_deterministic](https://www.tensorflow.org/api_docs/python/tf/data/Options?version=nightly#experimental_deterministic),
which may give a slight performance boost.

### Faster image decoding

By default TFDS automatically decode images. However, there are cases where it
can be more performant to skip the image decoding with
`tfds.decode.SkipDecoding` and manually apply the `tf.io.decode_image` op:

*   When filtering examples (with `ds.filter`), to decode images after examples
    have been filtered.
*   When cropping images, to use the fused `tf.image.decode_and_crop_jpeg(` op.

The code for both examples is available in the
[decode guide](https://www.tensorflow.org/datasets/decode#usage_examples).
