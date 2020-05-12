#include "mbed.h"
#include <cmath>
#include "DA7212.h"

#include "uLCD_4DGL.h"

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define bufferLength (32)

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];

DigitalOut led1(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);
InterruptIn sw2(SW2);
InterruptIn sw3(SW3);
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);

EventQueue queueDNN(32 * EVENTS_EVENT_SIZE);

Thread threadDNN(osPriorityNormal, 120*1024);

int songnow = 0, mode = 0, song_selection = 0;
int song[120];
int noteLength[120];
char serialInBuffer[bufferLength];
int serialCount = 0;
/*
int song[120] = {
  //Twinkle Twinkle Little Star
  261, 261, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294, 294, 261, 392, 392, 349, 349, 330, 330, 294, 392, 392, 349, 349, 330, 330, 294, 261, 261, 392, 392, 440, 440, 392, 349, 349, 330, 330, 294,           
  //Two-Tiger Run Fast
  524, 588, 660, 524, 524, 588, 660, 524, 660, 698, 784, 660, 698, 784, 784, 880, 784, 698, 660, 524, 784, 880, 784, 698, 660, 524, 588, 392, 524, 588, 392, 524, 524, 588, 660, 524, 524, 588, 660, 524,                
  //Little Bee
  392, 330, 330, 349, 294, 294, 262, 294, 300, 349, 392, 392, 392, 392, 330, 330, 349, 294, 294, 262, 330, 392, 392, 330, 392, 330, 330, 349, 294, 294, 262, 294, 300, 349, 392, 392, 392, 392, 330, 330};                    

int noteLength[120] = {
  //Twinkle Twinkle Little Star
  1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,                     
  //Two-Tiger Run Fast
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,                        
  //Little Bee
  1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2};                          
*/
// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void DNN(void) {
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

 // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
      if(mode == 1){
        if(gesture_index == 0 && song_selection == 0){
          if(songnow == 2){
            songnow = 0;  
          }else{
            songnow++;
          }
        }else if(gesture_index == 0 && song_selection == 1){
            songnow = 0;
            song_selection = 0;
            led1 = 1;
        }else if(gesture_index == 1 && song_selection == 0){
          if(songnow == 0){
            songnow = 2;
          }else{
            songnow--;
          }
        }else if(gesture_index == 1 && song_selection == 1){
          songnow = 1;
          song_selection = 0;
          led1 = 1;
        }else if(gesture_index == 2 && song_selection == 0){
          song_selection = 1;
          led1 = 0;
        }else if(gesture_index == 2 && song_selection == 1){
          songnow = 2;
          song_selection = 0;
          led1 = 1;
        }else{
          uLCD.locate(1, 1);
          uLCD.printf("\nError             \n                  \n");
        }
      }
    }
  }
}

void loadSignal(void)
{
  led2 = 0;
  int i = 0, j = 0;
  float freq = 0, length = 0;
  serialCount = 0;
  audio.spk.pause();
  uLCD.locate(1, 1);
  uLCD.printf("\nLoad Signal       \n                  \n");
  while(i < 120)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        freq = (float) atof(serialInBuffer);
        song[i] = freq * 1000;
        serialCount = 0;
        i++;
      }
    }
  }
  while(j < 120)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        length = (float) atof(serialInBuffer);
        noteLength[j] = length * 1000;
        serialCount = 0;
        j++;
      }
    }
  }
  led2 = 1;
}

void mode_selection(void){
  mode = 1;
}

void confirm_selection(void){
  mode = 0;
}

void uLCDprint(void){
  if(songnow == 0){
    uLCD.locate(1, 1);
    uLCD.printf("\nTwinkle Twinkle   \nLittle Star       \n");
  }else if(songnow == 1){
    uLCD.locate(1, 1);
    uLCD.printf("\nTwo-Tigers        \n                  \n");
  }else if(songnow == 2){
    uLCD.locate(1, 1);
    uLCD.printf("\nLittle Bee        \n                  \n");
  }else{
    uLCD.locate(1, 1);
    uLCD.printf("\nError             \n                  \n");
  }
}

void playNote(int freq) {
  for(int i = 0; i < kAudioTxBufferSize; i++){
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

int main(int argc, char* argv[]) {
  led1 = 1;
  led2 = 1;
  led3 = 1;
  threadDNN.start(DNN);
  sw2.rise(mode_selection);
  sw3.rise(confirm_selection);

  loadSignal();

  while(1){
    uLCDprint();
    for(int i = 0; i < 40; i++){
      if(mode == 0){
        int length = noteLength[40 * songnow + i];
        while(length--){
        // the loop below will play the note for the duration of 1s
          for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j){
            playNote(song[40 * songnow + i]);
          }
        }
      }else{
        audio.spk.pause();
      }
    }
  }

}