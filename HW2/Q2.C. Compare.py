#test data using rbf
x_test = np.arange(-200, 200) / 50
rbf_predictions = rbf.predict(x_test)


#test data using mlp
mlp_predictions = model.predict(x_test)


plt.plot(x_test, mlp_predictions)
plt.plot(x_test, rbf_predictions)
plt.plot(x_test, np.sin(x_test))
plt.legend(['MLP', 'RBF', 'sin'])
plt.show()



