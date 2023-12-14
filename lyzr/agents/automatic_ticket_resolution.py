import pandas as pd

def download_files(bucket_name):
    import os
    import boto3
    from botocore.handlers import disable_signing

    s3 = boto3.client("s3")
    s3.meta.events.register("choose-signer.s3.*", disable_signing)
    files = s3.list_objects_v2(Bucket=bucket_name)["Contents"]
    for file in files:
        file_path = file["Key"]
        local_file_path = "/content/" + file_path
        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path))
        s3.download_file(bucket_name, file_path, local_file_path)


def load_model(model_path):
    from tensorflow.keras.models import load_model as keras_load_model

    download_files("uniliver-demo-model")
    return keras_load_model(model_path)

class LyzrAutomaticTicketResolutionSystem:
    def __init__(self, model_path, client, resolution_models):
        self.model = load_model(model_path)
        self.client = client
        self.resolution_models = resolution_models
        self.rlhf = pd.DataFrame(
            {
                "ticket": pd.Series(dtype="str"),
                "response_1": pd.Series(dtype="str"),
                "response_2": pd.Series(dtype="str"),
                "best_response": pd.Series(dtype="int"),
            }
        )

    def load_model(model_path):
        from tensorflow.keras.models import load_model as keras_load_model

        return keras_load_model(model_path)

    def load_model_from_s3(model_path):
        from tensorflow.keras.models import load_model as keras_load_model

        download_files("uniliver-demo-model")
        return keras_load_model(model_path)

    def _predict(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        probabilities = self.model.predict(texts)
        predictions = [1 if prob > 0.5 else 0 for prob in probabilities.flatten()]
        return ["L1" if p == 0 else "L2" for p in predictions]

    def classify_tickets(self, tickets):
        classified_tickets = self._predict(tickets)
        for i in range(len(tickets)):
            print(
                f"Ticket: {tickets[i]}\nSuccessfully Classified as {classified_tickets[i]}"
            )

    def _resolve_ticket(self, ticket):
        ticket_category = self._predict([ticket])[0]

        if ticket_category in self.resolution_models:
            model_id = self.resolution_models[ticket_category]
            response = self.client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": ticket}], n=2
            )
            print(f"Ticket Successfully classified as {ticket_category}")
            print("Here are the resolutions")
            for i in range(len(response.choices)):
                print(f"Response {i+1}:", response.choices[i].message.content)
            print("Please Provide Feedback, for training purposes")

            best_resolution = int(input("Best suitable resolution (1/2): "))
            return (
                response.choices[0].message.content,
                response.choices[1].message.content,
                best_resolution,
                ticket,
            )

        else:
            print(f"No resolution model found for category: {ticket_category}")
            return None, None, None, ticket

    def resolution(self, tickets):
        for ticket in tickets:
            output1, output2, best_output, resolved_ticket = self._resolve_ticket(
                ticket
            )
            new_row = {
                "ticket": resolved_ticket,
                "response_1": output1,
                "response_2": output2,
                "best_response": best_output,
            }
            self.rlhf.loc[len(self.rlhf)] = new_row

    def prepare_data(self, df, label):
        import json

        filtered_df = df[df["Type"] == label]

        formatted_messages = []
        for _, row in filtered_df.iterrows():
            formatted_message = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Smart Ticket Resolution System",
                    },
                    {"role": "user", "content": row["user"]},
                    {"role": "assistant", "content": row["assistant"]},
                ]
            }
            formatted_messages.append(json.dumps(formatted_message))

        json_filename = f"/content/{label.lower()}_formatted_messages.jsonl"

        with open(json_filename, "w") as file:
            for message in formatted_messages:
                file.write(message + "\n")

        return json_filename

    def finetune_resolution_model(self, filename, model_name="gpt-3.5-turbo-1106"):
        with open(filename, "rb") as file:
            file_data = self.client.files.create(file=file, purpose="fine-tune")

        train_job = client.fine_tuning.jobs.create(
            training_file=file_data.id, model=model_name
        )
        return train_job

    def train_classifier(
        self, df, use_url="https://tfhub.dev/google/universal-sentence-encoder/4"
    ):
        from tensorflow.keras.utils import to_categorical
        import tensorflow as tf
        import tensorflow_hub as hub
        from tensorflow_hub import KerasLayer
        from tensorflow.keras.models import Model
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras.layers import Dense, Input, Dropout
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df["Type"])
        y = to_categorical(integer_encoded)
        num_classes = y.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            df["Text"], y, test_size=0.2, random_state=42
        )

        input_text = Input(shape=[], dtype=tf.string)
        embedding = hub.KerasLayer(use_url, trainable=True)(input_text)
        dense1 = Dense(256, activation="relu")(embedding)
        dropout = Dropout(0.2)(dense1)
        output = Dense(num_classes, activation="softmax")(dropout)
        model = Model(inputs=input_text, outputs=output)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, verbose=1, restore_best_weights=True
        )
        callbacks_list = [early_stopping]
        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            validation_data=(X_test, y_test),
            batch_size=8,
            callbacks=callbacks_list,
        )

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
        y_pred_probs = model.predict(X_test)
        y_pred = y_pred_probs.argmax(axis=-1)
        y_test_labels = y_test.argmax(axis=-1)

        cm = confusion_matrix(y_test_labels, y_pred)
        report = classification_report(
            y_test_labels, y_pred, target_names=label_encoder.classes_
        )
        print(report)
        model.save("/content/saved_model/my_model")

        return model, history, loss, accuracy
