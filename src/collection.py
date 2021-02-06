from milvus import Milvus, MetricType

# Milvus server IP address and port.
# Because the link to milvus in docker-compose 
# was named `milvus`, thats what the hostname will be
_HOST = 'localhost'
_PORT = '19530'  # default value

milvus = Milvus(_HOST, _PORT, pool_size=10)

"""
Get collection if not exists then create new one
for `collection` name of collection (string)
for `dimension` is length of vector (int)
for `index_file_size` is max file size of stored index (int)
"""
def make_collection(collection="simple", dimension=512, index_file_size=32):
    status, ok = milvus.has_collection(collection)

    if ok: return

    params = {
        'collection_name': collection,
        'dimension': dimension,
        'index_file_size': index_file_size,  # optional
        'metric_type': MetricType.L2  # optional
    }

    milvus.create_collection(params)

    return milvus
