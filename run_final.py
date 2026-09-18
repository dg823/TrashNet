import os
import subprocess
import time

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"🚀 단계: {description}")
    print(f"{'='*60}")
    print(f"실행 명령어: {command}\n")
    
    start_time = time.time()
    
    # Run the command and stream output
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if process.returncode == 0:
        print(f"\n✅ 완료 (소요 시간: {elapsed_time:.2f}초)")
    else:
        print(f"\n❌ 오류 발생! (코드: {process.returncode})")
        exit(1)

if __name__ == "__main__":
    print("🌟 EcoSortNet 최종 파이프라인 실행 🌟\n")
    print("이 스크립트는 다음 과정을 순서대로 자동 실행합니다:")
    print("1. 최적 설정으로 모델 학습 (60 Epochs, LR=0.0001, Augmentation 적용)")
    print("2. 결과 분석 및 시각화 (Confusion Matrix, Misclassified Examples 저장)")
    print("3. 샘플 이미지 추론 시연\n")
    
    # 1. 학습 실행
    # main.py 기본값이 이미 epochs=60, lr=0.0001로 설정되어 있음
    run_command("python main.py --mode train", "최종 모델 학습 시작")
    
    # 2. 결과 분석 및 보고서 생성
    run_command("python main.py --mode analyze", "성능 분석 및 그래프 생성")
    
    # 3. 추론 시연 (테스트용 이미지가 있다면)
    # data/cardboard 폴더의 첫 번째 이미지를 샘플로 사용해 봅니다.
    sample_img_path = "data/cardboard/cardboard1.jpg"
    
    if os.path.exists(sample_img_path):
        run_command(f"python main.py --mode infer --image {sample_img_path}", "샘플 이미지 분류 및 분리배출 가이드 테스트")
    else:
        print(f"\n⚠️ 샘플 이미지를 찾을 수 없어 추론 테스트는 건너뜁니다. ({sample_img_path})")
    
    print("\n🎉 모든 파이널 파이프라인이 성공적으로 완료되었습니다!")
    print("👉 생성된 분석 결과는 'results' 폴더에서 확인하실 수 있습니다.")
