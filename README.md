# Tumor detection
Нейросеть на основе inception_resnet_v2 для определения наличия опухоли головного мозга на снимках МРТ

# Датасет

![dataset](https://github.com/mosvlad/tumor_detection/blob/main/images/DoUHOmFenJA.jpg?raw=true "Dataset")

# Архитектура сети

Сеть построена на основе inception_resnet_v2 предобученном на Image Net датасете. В конец сети добавлен полносвязный слой, dropout и batch_norm, а выходом является 1 нейрон с softmax. В итоге на выходном нейроне будем иметь вероятность наличия опухоли на снимке. 

![architecture](https://github.com/mosvlad/tumor_detection/blob/main/images/model_plot.png?raw=true "architecture")

# Обучение 

Модель обучалась со следующими параметрами на протяжении 40 эпох

* loss='binary_crossentropy'
* optimizer=Adam(learning_rate=0.001)
* metrics=['accuracy']

![image](https://user-images.githubusercontent.com/31764930/146544239-08ac2beb-797b-4081-b40f-8c839f4efba2.png)

# Результаты

![image](https://user-images.githubusercontent.com/31764930/146544623-31b63660-31b8-4736-ada0-a5bff32debaa.png)
![image](https://user-images.githubusercontent.com/31764930/146544633-7ffd16b3-9c7b-4fe8-b6fb-e2f2bcf5e86a.png)

# Как запустить?

Проект обучался и тестировался на windows 10 с python3.7

```
git clone https://github.com/mosvlad/tumor_detection
cd tumor_detection
pip install -r requirements.txt

python main.py
```

В файле main.py 2 основных опции :
* Для запуска процесса обучния необходимо раскомментировать

```
trainer = train.TumorDetectionNet()
trainer.train(path_to_dataset="archive/", model_filename="Tumor_classifier_model.h5") 
```

* Для запуска проекта

```
evaluator = evaluate.Evaluator()
evaluator.evaluate(model_path="Tumor_classifier_model_v2.h5", image_path="archive/validation_data/323.jpg")
```

Веса предобученной сети доступны по ссылке :[Google Drive](https://drive.google.com/file/d/1Uatua4sb1Tzct-Ou4SNMIBBBbFAJmIpt/view?usp=sharing)
