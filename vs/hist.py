def plothist_2groups(X_no, X_etoh, bins, feature_name):
    plt.hist(X_no, bins, alpha=0.5, label='no ethanol')
    plt.hist(X_etoh, bins, alpha=0.5, label='ethanol')
    plt.title(feature_name)
    plt.legend(loc='upper right')
    plt.show()