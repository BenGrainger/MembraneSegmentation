from gunpowder.torch import Predict
import gunpowder as gp


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



class predict_pipeline(object):
    def __init__(self, source, model, inputs, pred_outs, input_size, output_size, checkpoint):
        """ create a predict pipeline
        Args:

            source:

            model: pytorch model

            inputs: (dict) dictionary of iterated predicted gp.Arraykeys placeholders depending on the model e.g. {0: raw}

            pred_outs: (dict) dictionary of iterated predicted gp.Arraykeys placeholders depending on the model e.g. {0: pred_lsds, 1; pred_affs}

            checkpoint: (str)

            input_size: 
        """
        self.source = source
        self.model = model
        self.inputs = inputs
        self.pred_outs = pred_outs
        self.checkpoint = checkpoint
        self.input_size = input_size
        self.output_size = output_size

    def create_pipeline(self):
        
        # request prediction batch
        scan_request = gp.BatchRequest()

        for i in self.inputs.values():
            scan_request.add(i, self.input_size)

        for i in self.pred_outs.values():
            scan_request.add(i, self.output_size)

        # set model to eval mode
        self.model.eval()

        # add a predict node
        predict = self.return_predict_node()

        scan = gp.Scan(scan_request)

        pipeline = self.source
        pipeline += gp.Normalize(self.inputs.values()[0])

        # raw shape = h,w

        pipeline += gp.Unsqueeze([self.inputs.values()[0]])

        # raw shape = c,h,w

        pipeline += gp.Stack(1)

        # raw shape = b,c,h,w

        pipeline += predict
        pipeline += scan

        # raw shape = c,h,w
        # pred shape = b,c,h,w

        squeeze_inputs = [i for i in self.inputs.values()] + [i for i in self.pred_outs.values()]
        pipeline += gp.Squeeze(squeeze_inputs)

        # raw shape = h,w
        # pred shape = c,h,w

        return pipeline

    def return_predict_node(self):
        predict = Predict(
            model=self.model,
            checkpoint=self.checkpoint,
            inputs = self.inputs,
            outputs = self.pred_outs)
        
        return predict



