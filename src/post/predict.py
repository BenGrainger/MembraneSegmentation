from gunpowder.torch import Predict
import gunpowder as gp

def predict_node(model, raw, pred_outs, checkpoint):
    """ create a predict node for MTLSD models
    Args:

        raw: (gp.Arraykey)

        pred_outs: (dict) dictionary of iterated predicted gp.Arraykeys placeholders depending on the model e.g. {0: pred_lsds, 1; pred_affs}

        checkpoint: (str)
    """
    predict = Predict(
        model=model,
        checkpoint=checkpoint,
        inputs = {
                'input': raw
        },
        outputs = pred_outs)
    
    return predict


def get_input_output_roi(source, raw, input_size, output_size):
    """ in order to scan over the entire dataset the total input and output sizes must be known - these are returned here
    Args:

        source:

        raw: (gp.Arraykey)

        input_size: (gp.Coordinate)

        output_size: (gp.Coordinate)
    """
    context = (input_size - output_size) / 2

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context,-context)
    
    return total_input_roi, total_output_roi


def predict_pipeline(source, model, raw, pred_outs, input_size, output_size, checkpoint):
    """ create prediction pipeline
    Args:

        source:
        
        model:

        raw:

        pred_outs: (dict) dictionary of iterated predicted gp.Arraykeys placeholders depending on the model e.g. {0: pred_lsds, 1; pred_affs}

        input_size:

        output_size:

        checkpoint: 
    """
    # request prediction batch
    scan_request = gp.BatchRequest()

    scan_request.add(raw, input_size)

    for i in pred_outs.values():
        scan_request.add(i, output_size)

    # set model to eval mode
    model.eval()

    # add a predict node
    predict = predict_node(model, raw, pred_outs, checkpoint)

    scan = gp.Scan(scan_request)

    pipeline = source
    pipeline += gp.Normalize(raw)

    # raw shape = h,w

    pipeline += gp.Unsqueeze([raw])

    # raw shape = c,h,w

    pipeline += gp.Stack(1)

    # raw shape = b,c,h,w

    pipeline += predict
    pipeline += scan
    pipeline += gp.Squeeze([raw])

    # raw shape = c,h,w
    # pred shape = b,c,h,w

    squeeze_inputs = [raw] + [i for i in pred_outs.values()]
    pipeline += gp.Squeeze(squeeze_inputs)

    # raw shape = h,w
    # pred shape = c,h,w

    return pipeline