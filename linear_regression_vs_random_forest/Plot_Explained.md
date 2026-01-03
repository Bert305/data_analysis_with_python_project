




Blue dots = Linear Regression predicted values
Red dots = Random Forest predicted values
Diagonal dashed line = A reference line showing where points would fall if predictions were perfect (where predicted value = actual value)
The actual values are represented by the x-axis position of each dot.

So for each dot:

X-coordinate = actual logS value from y_test
Y-coordinate = predicted logS value from the model


The diagonal line is just a visual guide: if a model predicted perfectly, the dot would land on that line because predicted would equal actual. The farther a dot is from the diagonal line, the worse that prediction was.


For a dot at coordinates (-8, -8.2):

x = -8 is the actual logS value from the dataset
y = -8.2 is what the model predicted
The prediction was off by 0.2, which is why the dot is slightly below the diagonal line.

If the prediction were perfect, the dot would be at (-8, -8) right on the diagonal line.

The closer dots cluster to the diagonal, the better the model's predictions are.