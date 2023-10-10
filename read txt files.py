import pandas as pd

with open("орион.txt") as doc:
    contents = doc.readlines()
    result = []
    for i in contents:
        result.append(i.replace("\n", ""))
    result_data = {'names': result}
    df = pd.DataFrame(result_data)