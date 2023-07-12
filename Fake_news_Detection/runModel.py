import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

dataset, info = tfds.load('imdb_reviews', with_info=True,as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

def predict_prob(dataset):
    review_type = []
    loaded_model = tf.keras.models.load_model('saved_Model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

    for review, label in dataset.batch(1128):
        predictions = loaded_model.predict(review)
        for i in range(len(predictions)):
            pos_review = predictions[i][0]
            neg_review = 1 - predictions[i][0]

            if pos_review > neg_review:
                sentiment = 'Positive'
                confidence = pos_review
            else:
                sentiment = 'Negative'
                confidence = neg_review

            review_type.append({
                'Review': review[i].numpy().decode('utf-8'),
                'Sentiment': sentiment,
                'Confidence': confidence
            })

    return review_type

output=predict_prob(test_dataset)
#Test One example from the output
print()
print("The Review : \n",output[0]["Review"])
print()
print("The Sentiment : \n",output[0]["Sentiment"])
print()
print("The Confidence : \n",output[0]["Confidence"])