# StealthFlow

## ResNeSt (TFlayer)

x = ResNeStBlock(radix=2, cardinality=2, bottleneck=64, ratio=4)(x)

## FID (4-D Tensor x2 -> FID score)

fid_score = FIDNumpy(batch_size=50, scaling=True)(images1, images2)
fid_score = FIDTF(batch_size=50, scaling=True)(images1, images2)
