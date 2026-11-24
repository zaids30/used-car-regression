import numpy as np

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

cars = pd.read_csv("data/used_car_sales.csv")
lots = pd.read_csv("data/auction_lots.csv")
cars["Accident"] = (cars["Accident"] == "yes").astype(int)
lots["Accident"] = (lots["Accident"] == "yes").astype(int)

# 70/30 split
test = cars.sample(frac=0.3, random_state=42)
train = cars.drop(test.index)
print(len(train), "train,", len(test), "test")

# full model vs model without Num_Owners
full = smf.ols("Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident + Num_Owners", train).fit()
v1 = smf.ols("Sale_Price ~ Mileage_K + Age_Years + Horsepower + Accident", train).fit()
print(full.summary())
print(v1.summary())

# baseline vs v1 on test cars
avg = train["Sale_Price"].mean()
y = test["Sale_Price"]
pred = v1.predict(test)

def r2(actual, predicted):
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()

def rmse(actual, predicted):
    return np.sqrt(((actual - predicted) ** 2).mean())

print("baseline: R2", round(r2(y, avg), 3), "RMSE", round(rmse(y, avg)))
print("v1:       R2", round(r2(y, pred), 3), "RMSE", round(rmse(y, pred)))

# buy rule on the 40 auction lots
def buy_or_pass(car, auction_price):
    predicted = v1.predict(pd.DataFrame([car])).iloc[0]
    if auction_price + 1500 < predicted:
        return "buy"
    return "pass"

lots["v1"] = [buy_or_pass(row, row["Auction_Price"]) for _, row in lots.iterrows()]
lots["baseline"] = np.where(lots["Auction_Price"] + 1500 < avg, "buy", "pass")
lots["profit"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - 1500

for rule in ["baseline", "v1"]:
    bought = lots[lots[rule] == "buy"]
    print(rule, "- bought:", len(bought), "losers:", (bought["profit"] < 0).sum(), "profit:", bought["profit"].sum())

# predicted vs actual chart
plt.scatter(y, pred, label="v1")
plt.axhline(avg, color="orange", linestyle="--", label="baseline (average)")
plt.plot([0, 45000], [0, 45000], color="grey", label="perfect prediction")
plt.xlabel("Actual sale price ($)")
plt.ylabel("Predicted sale price ($)")
plt.title("Test cars: predicted vs actual")
plt.legend()
plt.savefig("predicted_vs_actual.png")
