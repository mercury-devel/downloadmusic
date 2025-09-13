from pyrogram import Client

api_id = 25983686
api_hash = "d49ffa3e2b617c66250b7f4c169d1cb9"
chat_id = 'homaokla'

app = Client(
    "murderculture",
    api_id=api_id,
    api_hash=api_hash
)

app.start()
while True:
    try:
        msgs = [message.id for message in app.get_chat_history(chat_id, limit=100)]
        print(msgs)
        app.delete_messages(chat_id, msgs)
    except Exception as e:
        print(e)
app.stop()
