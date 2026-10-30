# Classification parameters

`ACDC.pt` is a tensor-only state_dict: 127 tensors, 4,538,375 parameters. Use the root config.json and classes.json with infer.py. It contains no optimizer, patient lists or training metadata.

The current parameters replace the earlier classification model. They come from a local exploratory run where 21 of the 30 test-partition patients participated in optimization and nine did not receive gradient updates. The checkpoint was selected using scores on the mixed 30-patient partition. Its accuracy is 80.00% on the mixed partition and 33.33% on the reserved nine patients. These are not independent-test or paper-reproduction results. Validation details and the file SHA-256 are recorded in ../VALIDATION.json.
