import numpy as np
import pandas as pd
import bentoml
from bentoml.io import JSON

model_ref = bentoml.sklearn.get("wine:latest")

scaler = model_ref.custom_objects['scaler']

model_runner = model_ref.to_runner()

svc = bentoml.Service("wine", runners=[model_runner])

def pipeline_process(application_data):
    print(application_data)
    df = pd.DataFrame.from_dict([application_data])
    print(df)

    # Scale the test set
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df

@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    df = pipeline_process(application_data)

    print(df)
    prediction = await model_runner.predict.async_run(df)
    print(prediction)
    result = prediction[0].round().clip(0, 10)

    return {
        "quality": result
    }
