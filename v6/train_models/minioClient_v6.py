from minio import Minio
from minio import ResponseError
import os
# Create client Minio
## Address IP, access_k, secret_key as arguments
## return minio client
def init_minio(address,access_k,secret_k):
    print('{}: init_minio function'.format(os.path.basename(__file__)))
    minioCl = Minio(address,
        access_key = access_k,
        secret_key = secret_k,
        secure = False) # Secure = True not working
    return minioCl


# Display buckets list
## Minio client as argument
## Display buckets list
## return buckets list
def get_buckets(minioCl):
    print('{}: get_buckets function'.format(os.path.basename(__file__)))
    bucket_list = minioCl.list_buckets()

    for bucket in bucket_list:
        print(bucket.name, bucket.creation_date)
    return bucket_list

'''
# get object
try:
    response = minioCl.get_object('ships', 'shipsnet.json')
    print('response type: {0}'.format(type(response)))
    print('response : {0}'.format(response))
    
finally:
    response.close()
    response.release_conn()

# Get object and save in local
try:
    resp = minioCl.fget_object('ships','shipsnet.json','datas/shipsnet.json')
finally:
    response.close()
    response.release_conn()
'''

