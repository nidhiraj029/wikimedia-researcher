import pyarrow.parquet as pq

path = "data/raw/enwiki_namespace_0_00009.parquet"

parquet = pq.ParquetFile(path)

print("Rows:", parquet.metadata.num_rows)
print("Row groups:", parquet.num_row_groups)

print("\n===== COLUMNS =====")

for column in parquet.schema.names:
    print("-", column)

print("\n===== SCHEMA =====")
print(parquet.schema)

print("\n===== SAMPLE =====")

table = parquet.read_row_group(0)

record = table.to_pylist()[0]

for key, value in record.items():
    print(f"\n{key}:")
    print(str(value)[:500])