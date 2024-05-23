import json
import urllib.parse
import boto3
import subprocess

print('Loading function')

s3 = boto3.client('s3')
lmbda = boto3.client('lambda')
    

def video_splitting_cmdline(vid_path, img_path):
    print("inside the splitting method")
    print(f"image path: {img_path}")
    print(f"video path: {vid_path}")

    split_cmd = f'./ffmpeg -i "{vid_path}" -vframes 1 "{img_path}"'

    try:
        print("calling the ffmpeg command using subprocess")
        subprocess.check_call(split_cmd, shell=True)

    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)



def lambda_handler(event, context):
    input_bucket = event['Records'][0]['s3']['bucket']['name']
    output_bucket = '1225464032-stage-1'
    
    input_key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        print("fetching video from input bucket")
        response = s3.get_object(Bucket=input_bucket, Key=input_key)
        print("successfully fetched video")

        vid_path = f"/tmp/{input_key}"
        with open(vid_path, 'wb') as f:
            f.write(response['Body'].read())
        print(f"stored video locally at {vid_path}")
            
        image_name = f"{input_key.split('.')[0]}.jpg"
        img_path = f"/tmp/{image_name}"
        print(f"image_path: {img_path}")
        
        print("calling the splitting method")
        video_splitting_cmdline(vid_path, img_path)

        s3_key = image_name
        print(f"s3 key: {s3_key}")
        s3.upload_file(img_path, output_bucket, s3_key)
        print(f"Uploaded {img_path} to s3://{output_bucket}/{s3_key}")

        print('invoking face recognition function')
        inputParams = {
            "bucket_name" : "1225464032-stage-1",
            "image_file_name" : image_name
        }
        response = lmbda.invoke(
            FunctionName = 'arn:aws:lambda:us-east-1:992382850355:function:face-recognition',
            InvocationType = 'Event',
            Payload = json.dumps(inputParams)
        )
 
        # responseFromChild = json.load(response['Payload'])
        # print(responseFromChild)
        
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(input_key, input_bucket))
        raise e