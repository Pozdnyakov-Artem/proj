import telebot
from dotenv import load_dotenv
import os

from pathlib import Path
Path("subscribers.txt").touch()

load_dotenv()
bot=telebot.TeleBot(os.getenv("TELEGRAM_API_KEY"))

def read_subscribers(file):
    text=file.read()
    if len(text)== 0:
        return []
    return list(map(int,text.split(',')))


def write_subscribers(file,subscribers):
    subscribers=list(map(str,subscribers))
    file.write(','.join(subscribers))
    
    
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Добавлен в подписку.\n/unsubscribe для отмены")
    with open("subscribers.txt","r") as f:
        subscribers=read_subscribers(f)
    if(message.from_user.id not in subscribers):
        subscribers.append(message.chat.id)
    with open("subscribers.txt","w") as f:
        write_subscribers(f,subscribers)
        
        
@bot.message_handler(commands=['unsubscribe'])
def unsubscribe(message):
    with open("subscribers.txt","r") as f:
        subscribers=read_subscribers(f)
    try:
        subscriber_index=subscribers.index(message.chat.id)
    except:
        return
    subscribers.pop(subscriber_index)
    with open("subscribers.txt","w") as f:
        write_subscribers(f,subscribers)
    bot.reply_to(message,"Успешно")
        
        
def send_alert(cam_id,n_objects=1,classes=[]):
    with open("subscribers.txt","r") as f:
        subscribers=read_subscribers(f)
    message=f"ОБНАРУЖЕНИЕ\nНа камере '{cam_id}'\nКоличество: {n_objects}\n"
    if len(classes) > 0:
        message += ','.join(list(map(str,classes)))
    for subscriber in subscribers:
        bot.send_message(subscriber,message)
if __name__ == "__main__":
    bot.infinity_polling()
