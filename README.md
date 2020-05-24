# StealthFlow

## ResNeSt (TFlayer)

see (JP): https://note.com/hyper_pigeon/n/n12988580739d

x = ResNeStBlock(radix=2, cardinality=2, bottleneck=64, ratio=4)(x)

## FID (4-D Tensor x2 -> FID score)

see (JP): https://note.com/hyper_pigeon/n/n9c5643413cd7

### Numpy

fid_score = FIDNumpy(batch_size=50, scaling=True)(images1, images2)

### TF

fid_score = FIDTF(batch_size=50, scaling=True)(images1, images2)
