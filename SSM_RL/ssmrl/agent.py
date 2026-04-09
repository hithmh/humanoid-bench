import numpy as np
import tensorflow as tf
import os


def detect_nan_in_dict(x):
    for key in x.keys():
        if np.isnan(x[key]):
            return True
    return False

class base_agent(tf.keras.Model):

    def __init__(self, args, *kwargs):
        super().__init__()
        self.diagnotics = {}
        self.opt_list = []
        self.evaluation_diagnotics = {}
        self.args = args
        self._create_place_holders()
        self._build_model()
        self._create_optimizer()
        self._build_controller()

        # Use checkpoint instead of Saver
        self.checkpoint = tf.train.Checkpoint(model=self)


    def _build_model(self):

        pass

    def _build_controller(self):

        pass

    def _create_optimizer(self):

        pass

    def _create_place_holders(self):

        self.act_dim = self.args['act_dim']

        # Initialize shift and scale variables
        self.shift_holder = tf.Variable(np.zeros(self.args['state_dim']), trainable=False, name="state_shift", dtype=tf.float32)
        self.scale_holder = tf.Variable(np.ones(self.args['state_dim']), trainable=False, name="state_scale", dtype=tf.float32)
        self.shift_u_holder = tf.Variable(np.zeros(self.args['act_dim']), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u_holder = tf.Variable(np.ones(self.args['act_dim']), trainable=False, name="action_scale", dtype=tf.float32)

        # Create Input layers (TF2 replacement for placeholders)
        # These can be used with Keras functional API
        if 'pred_horizon' in self.args:
            # For model-based agents with prediction horizon
            self.x_input = tf.keras.Input(
                shape=(self.args['pred_horizon'], self.args['state_dim']),
                dtype=tf.float32,
                name='x_input'
            )
            self.a_input = tf.keras.Input(
                shape=(self.args['pred_horizon']-1, self.args['act_dim']),
                dtype=tf.float32,
                name='a_input'
            )
            self.r_input = tf.keras.Input(
                shape=(self.args['pred_horizon'],),
                dtype=tf.float32,
                name='r_input'
            )
        else:
            # For model-free agents (single-step)
            self.x_input = tf.keras.Input(
                shape=(self.args['state_dim'],),
                dtype=tf.float32,
                name='x_input'
            )
            self.a_input = tf.keras.Input(
                shape=(self.args['act_dim'],),
                dtype=tf.float32,
                name='a_input'
            )
            self.r_input = tf.keras.Input(
                shape=(1,),
                dtype=tf.float32,
                name='r_input'
            )

        # Learning rate will be passed as argument to optimizer


    def set_shift_and_scale(self, replay_memory):
        self.shift_holder.assign(replay_memory.shift_x)
        self.scale_holder.assign(replay_memory.scale_x)
        self.shift_u_holder.assign(replay_memory.shift_u)
        self.scale_u_holder.assign(replay_memory.scale_u)

    def get_shift_and_scale(self):
        # In TF2 eager mode, variables can be accessed directly
        return (self.shift_holder.numpy(),
                self.scale_holder.numpy(),
                self.shift_u_holder.numpy(),
                self.scale_u_holder.numpy())

    # def choose_action(self, s, evaluation=False):
    #
    #     pass


    def calc_test_loss(self, replay_memory):

        batch_dict = replay_memory.get_test_data_in_batch()
        x = batch_dict['states']
        u = batch_dict['inputs']
        c = batch_dict['costs']

        # Call evaluation method directly (to be implemented in subclasses)
        diagnotics = self._evaluate(x, u, c)
        output = {}
        [output.update({key + '_test': value}) for (key, value) in zip(self.evaluation_diagnotics.keys(), diagnotics)]

        return output

    def calc_val_loss(self, replay_memory):

        # data for validation
        batch_dict = replay_memory.get_all_val_data()
        x = batch_dict['states']
        u = batch_dict['inputs']
        c = batch_dict['costs']

        # Call evaluation method directly (to be implemented in subclasses)
        diagnotics = self._evaluate(x, u, np.squeeze(c))
        output = {}
        [output.update({key + '_val': value}) for (key, value) in zip(self.evaluation_diagnotics.keys(), diagnotics)]

        return output

    def _evaluate(self, x, u, r):
        """Compute evaluation diagnostics - to be implemented in subclasses"""
        return [self.evaluation_diagnotics[key] for key in self.evaluation_diagnotics.keys()]

    def training_evaluation(self, replay_memory):

        output = {}
        val_eval_dict = self.calc_val_loss(replay_memory)
        [output.update({key: value}) for (key, value) in zip(val_eval_dict.keys(), val_eval_dict.values())]

        # test_eval_dict = self.calc_test_loss(replay_memory)
        # [output.update({key: value}) for (key, value) in zip(test_eval_dict.keys(), test_eval_dict.values())]

        return output

    def learn(self, replay_memory, lr):
        batch_dict = replay_memory.random_sample()
        x = batch_dict['states']
        u = batch_dict['inputs']
        c = batch_dict['costs']
        c = np.squeeze(c)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        u = tf.convert_to_tensor(u, dtype=tf.float32)
        c = tf.convert_to_tensor(c, dtype=tf.float32)
        # Perform training step (to be implemented in subclasses)
        output = self._train_step(x, u, c, lr)
        ## transform the contents of output into numpy
        for key in output.keys():
            output[key] = output[key].numpy()


        # # check if the output is nan
        # if detect_nan_in_dict(output):
        #     print('nan detected')

        return output

    def _train_step(self, x, u, r, lr):
        """Perform one training step - to be implemented in subclasses"""
        # This should compute diagnostics and apply optimizations
        diagnotics_values = [self.diagnotics[key] for key in self.diagnotics.keys()]
        output = {}
        [output.update({key: value}) for (key, value) in zip(self.diagnotics.keys(), diagnotics_values)]
        return output

    def save_result(self, path, verbose=False):

        os.makedirs(path + "/model", exist_ok=True)

        save_path = self.checkpoint.save(path + "/model/")

        if verbose is True:
            print("Save to path: ", save_path)

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path + '/model/')
        if model_file is None:
            success_load = False
            return success_load
        self.checkpoint.restore(model_file)
        success_load = True

        return success_load

    def reset_for_control(self):
        pass

    def store_cached_control_info(self):
        pass
def mlp(input, sizes, activation, output_activation=None, name="", regularizer=None, reuse=None):
    """Creates a multi-layered perceptron using Tensorflow.

    Args:
        sizes (list): The size of each of the layers.

        activation (function): The activation function used for the
            hidden layers.

        output_activation (function, optional): The activation function used for the
            output layers. Defaults to tf.keras.activations.linear.

        regularizer (function, optional): Regularizer used to prevent overfitting

        name (str, optional): A nameprefix that is added before the layer name. Defaults
            to an empty string.

    Returns:
        output(Tensor): Output of the multi-layer perceptron
    """
    trainable = True if reuse is None else False

    # Create model using Keras layers
    for j in range(len(sizes) - 1):
        input = input if j == 0 else output
        act = activation if j < len(sizes) - 2 else output_activation

        layer = tf.keras.layers.Dense(
            sizes[j + 1],
            activation=act,
            name=name + "_l{}".format(j + 1) if name else "l{}".format(j + 1),
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            trainable=trainable
        )
        output = layer(input)
        # output = tf.keras.layers.Dropout(0.3)(output)
    return output










