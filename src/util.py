def window_average(a, window=3):
    csum = 0
    average = []
    for i in range(len(a)):
        csum += a[i]
        if i >= window:
            csum -= a[i - window]
        average.append(csum / min(i+1, window))
    return average
