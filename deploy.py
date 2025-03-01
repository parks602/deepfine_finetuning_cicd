from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import csv
import logging
import time
import sys

def find_best_model(csv_file):
    best_model = None
    highest_accuracy = 0.0
    
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            model_name = row['Model Name']
            accuracy = float(row['Accuracy'])
            if accuracy > highest_accuracy:
                best_model = model_name
                highest_accuracy = accuracy
    
    return best_model

def deploy_model(model_name):
    # 모델 파일의 경로
    global logger
    model_path = "./models/%s"%(model_name)  # 예시: 모델명.h5 형태의 파일
    
    # 배포할 경로
    logger.info("모델 %s을(를) 배포하는 중..."%(model_name))
    try:
        # 모델 파일을 배포할 경로로 복사
        return FileResponse(model_path, media_type="application/octet-stream", filename=model_name)
        logger.info("모델 %s을(를) 성공적으로 배포했습니다."%(model_name))
    except FileNotFoundError:
        logger.error("모델 %s 파일을 찾을 수 없습니다."%(model_name))
        raise HTTPException(status_code=404, detail="모델 파일을 찾을 수 없습니다.")
    except Exception as e:
        logger.error("모델  %s 배포 중 오류가 발생했습니다: %s}"%(model_name, str(e)))
        raise HTTPException(status_code=500, detail="모델 배포 중 오류가 발생했습니다: %s"%(str(e)))


app = FastAPI()
# 로거 생성
logger = logging.getLogger("deploy")
logger.setLevel(logging.DEBUG)

# 파일 핸들러 추가
file_handler = logging.FileHandler('deployment_logs.log',  encoding='utf-8', mode='w')
file_handler.setLevel(logging.DEBUG)
# 로그 포맷 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)
logger.addHandler(stream_hander)

# 로거에 파일 핸들러 추가
logger.addHandler(file_handler)
logger.info("서버가 실행되었습니다.")
csv_file_path = "./evaluator.csv"

print(find_best_model(csv_file_path))

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/deploy")
async def deploy_best_model():
    # CSV 파일 경로
    global logger
    csv_file_path = "./evaluator.csv"
    logger.info("START")
    # 가장 높은 정확도를 가진 모델명 찾기
    best_model_name = find_best_model(csv_file_path)

    if best_model_name:
        # 가장 높은 정확도를 가진 모델 배포
        success = deploy_model(best_model_name)
        if success:
            logger.info("%s 모델을 성공적으로 배포했습니다."%(best_model_name))
            return {"message": "%s 모델을 성공적으로 배포했습니다."%(best_model_name)}
            
        else:            
            logger.error("%s 모델 파일을 찾을 수 없습니다."%(best_model_name))
            raise HTTPException(status_code=404, detail="'%s 모델 파일을 찾을 수 없습니다."%(best_model_name))
    else:
        raise HTTPException(status_code=404, detail="CSV 파일에 모델 정보가 없거나 형식이 잘못되었습니다.")

# # 10초 후에 종료
# time.sleep(10)
# logger.info("서버가 10초 동안 실행되었습니다. 종료합니다.")
# sys.exit()