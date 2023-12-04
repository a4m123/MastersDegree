from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import cv2
import numpy as np
import model_architecture
import torch
from PIL import Image
import time

# Replace with the actual token you obtained from BotFather
TOKEN = '6737437100:AAFCEwHc_u_OOySatayvo2JfxoxM0tivD8g'

palette = {0: (255, 0, 0),  # alive
           3: (0, 255, 255),  # dying
           2: (255, 0, 255),  # recently dead
           1: (255, 255, 0),  # a long time ago dead
           4: (0, 0, 0)}  # background

invert_palette = {v: k for k, v in palette.items()}

# сегментация нейронной сети в RGB изображение
def convert_to_color(arr_2d, palette=palette):
   """ Numeric labels to RGB-color encoding """
   arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
   for c, i in palette.items():
      m = arr_2d == c
      arr_3d[m] = i
   return arr_3d

def predict_full_image(model_path="A:/Профиль/Загрузки/best_mIoU(11).pth",
                       input_image_path=''):
   
   # Load PyTorch model
   model = model_architecture.UNet()
   state_dict = torch.load(model_path)
   model.load_state_dict(state_dict)
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   model.to(device)
   model.eval()

   # Set the dimensions of the training fragments and full image
   fragment_size = (256, 256)

   # Load and preprocess the full image
   image = Image.open(input_image_path)
   image = np.array(image) / 255.0  # Normalize pixel values
   image = image[:,:,:3]
   # get full image size
   full_image_size = image.shape[:2]
   print(image.shape)
   image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
   # find the moment when after spliting full image we cannot get 256x256 fragments
   count_x = full_image_size[0] // fragment_size[0]
   count_y = full_image_size[1] // fragment_size[1]
   # get the size of the rightmost and bottommost fragments
   rightmost_fragment_size = (full_image_size[0] - count_x * fragment_size[0], fragment_size[1])
   bottommost_fragment_size = (fragment_size[0], full_image_size[1] - count_y * fragment_size[1])
   # get the size of the last fragment
   last_fragment_size = (full_image_size[0] - count_x * fragment_size[0], full_image_size[1] - count_y * fragment_size[1])
   # split full image into fragments. The last fragment of row will be shifted by the size of the rightmost fragment
   # The last fragment of column will be shifted by the size of the bottommost fragment
   fragments = []
   for i in range(0, full_image_size[0] - rightmost_fragment_size[0], fragment_size[0]):
      for j in range(0, full_image_size[1] - bottommost_fragment_size[1], fragment_size[1]):
         fragments.append(image[:, :, i:i + fragment_size[0], j:j + fragment_size[1]])
      fragments.append(image[:, :, i:i + fragment_size[0], -fragment_size[1]:])
   # add the last row of fragments
   for j in range(0, full_image_size[1] - bottommost_fragment_size[1], fragment_size[1]):
      fragments.append(image[:, :, -fragment_size[0]:, j:j + fragment_size[1]])
   # add the last fragment
   fragments.append(image[:, :, -fragment_size[0]:, -fragment_size[1]:])
   # convert list of fragments to tensor
   fragments = torch.cat(fragments, dim=0)
   # predict
   fragments = fragments.to(device)
   outputs = model(fragments)
   predicted = outputs.argmax(dim=1)
   predicted = predicted.cpu().numpy()
   # convert predicted fragments to full image
   predicted_image = np.zeros((full_image_size[0], full_image_size[1]))
   for i in range(0, full_image_size[0] - rightmost_fragment_size[0], fragment_size[0]):
      for j in range(0, full_image_size[1] - bottommost_fragment_size[1], fragment_size[1]):
         predicted_image[i:i + fragment_size[0], j:j + fragment_size[1]] = predicted[0]
         predicted = predicted[1:]
      predicted_image[i:i + fragment_size[0], -fragment_size[1]:] = predicted[0]
      predicted = predicted[1:]
   # add the last row of fragments
   for j in range(0, full_image_size[1] - bottommost_fragment_size[1], fragment_size[1]):
      predicted_image[-fragment_size[0]:, j:j + fragment_size[1]] = predicted[0]
      predicted = predicted[1:]
   # add the last fragment
   predicted_image[-fragment_size[0]:, -fragment_size[1]:] = predicted[0]
   # convert predicted image to color
   predicted_image = convert_to_color(predicted_image)
   print(predicted_image.shape)
   torch.cuda.empty_cache()
   return predicted_image

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! Send me an image, and I will send it back.')

def echo(update: Update, context: CallbackContext) -> None:
   start_time = torch.cuda.Event(enable_timing=True)
   end_time = torch.cuda.Event(enable_timing=True)
   # Check if the message contains a photo
   if update.message.photo:
      start_time.record()
      # Get the file_id of the first photo
      file_id = update.message.photo[-1].file_id
      try:
         # Get the File object using the file_id
         file_info = context.bot.get_file(file_id)
         # Download the file, predict the image
         file_path = file_info.download()
         predicted_image = predict_full_image(input_image_path=file_path)
         print(predicted_image.shape, np.unique(predicted_image))                      
         
         # Save the grayscaled image to a new file
         cv2.imwrite('result.jpg', predicted_image)
         # Send the image back to the user
         update.message.reply_photo(photo=open('result.jpg', 'rb'))
         end_time.record()
         torch.cuda.synchronize()
         print(start_time.elapsed_time(end_time)) # milliseconds

      except Exception as e:
          # Log any exceptions for debugging
          print("Error:", e)
          update.message.reply_text('Error processing the image. Please try again later.')
   else:
       update.message.reply_text('Please send me an image.')

def main() -> None:
    # Create the Updater and pass it your bot's token
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Register command and message handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you send a signal to stop it
    updater.idle()

if __name__ == '__main__':
    main()
