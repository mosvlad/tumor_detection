import train
import evaluate

if __name__ == "__main__":
    #trainer = train.TumorDetectionNet()
    #trainer.train(path_to_dataset="archive/", model_filename="Tumor_classifier_model.h5")

    evaluator = evaluate.Evaluator()
    evaluator.evaluate(model_path="Tumor_classifier_model_v2.h5", image_path="archive/validation_data/323.jpg")
