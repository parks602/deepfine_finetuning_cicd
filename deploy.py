from fastapi import FastAPI, HTTPException
import shutil
import csv

app = FastAPI()

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
    model_path = r"./models/%s"%(model_name)  # 예시: 모델명.h5 형태의 파일
    
    # 배포할 경로
    deploy_path = '/deployed_models'  # 예시: 배포할 디렉토리 경로
    
    try:
        # 모델 파일을 배포할 경로로 복사
        shutil.copy(model_path, deploy_path)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 배포 중 오류가 발생했습니다: {str(e)}")

@app.get("/deploy")
async def deploy_best_model():
    # CSV 파일 경로
    csv_file_path = r"./evaluator.csv"

    # 가장 높은 정확도를 가진 모델명 찾기
    best_model_name = find_best_model(csv_file_path)

    if best_model_name:
        # 가장 높은 정확도를 가진 모델 배포
        success = deploy_model(best_model_name)
        if success:
            return {"message": f"'{best_model_name}' 모델을 성공적으로 배포했습니다."}
        else:
            raise HTTPException(status_code=404, detail=f"'{best_model_name}' 모델 파일을 찾을 수 없습니다.")
    else:
        raise HTTPException(status_code=404, detail="CSV 파일에 모델 정보가 없거나 형식이 잘못되었습니다.")
