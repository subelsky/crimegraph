def test_model(model, data, batch_size, metrics):
    for m in metrics:
        m.reset_state()

    for x_test_batch, y_test_batch in data:
        predictions = model.predict(x_test_batch, verbose=0, batch_size=batch_size)

        for metric in metrics:
            metric.update_state(y_test_batch, predictions)

    results = {}

    for m in metrics:
        final_score = m.result().numpy()
        results[m.name] = final_score

        m.reset_state()
    
    return results