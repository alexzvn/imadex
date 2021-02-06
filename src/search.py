
from collection import milvus

def search(collection_name="simple", query_records=[], top_k=10):

    return milvus.search(**{
        'collection_name': collection_name,
        'query_records': query_records,
        'top_k': top_k,
        'params': { 'nprobe': 16 }
    })

