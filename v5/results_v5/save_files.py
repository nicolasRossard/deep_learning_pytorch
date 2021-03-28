import sys
sys.path.append('../')
import infos_v5 as infos
import minioClient_v5 as minioClient
minioCl = minioClient.init_minio(infos.ip_address,infos.access_key,infos.secret_key)
# Save files in Minio

minioCl.fput_object(infos.dir_json, 'results_v5.pkl', 'results_df.pkl')
minioCl.fput_object(infos.dir_json, 'Model_10_100_20_0001.pt', 'Model_10_100_20_0001.pt')
minioCl.fput_object(infos.dir_json, 'Optim_10_100_20_0001.pt', 'Optim_10_100_20_0001.pt')
        
