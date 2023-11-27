import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import openai
import pickle   
import time

# Đặt thời gian giữa các yêu cầu
WAIT_TIME_BETWEEN_REQUESTS = 1

last_request_time = time.time()


model = load_model('chatbot_model.h5')
intents = json.loads(open('job_intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

openai.api_key = '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'
engine = "text-davinci-003"  

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):

    if ints:
        tag = ints[0]['intent']
        # Xử lý khi danh sách ints rỗng
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
            else:
                result = "You must ask the right questions"
        return result
    else:
        result = "Sorry, I don't understand that."


# Đường dẫn đến tệp lưu trữ kết quả
CACHE_FILE_PATH = 'cache.pkl'

# Kiểm tra xem tệp lưu trữ đã tồn tại chưa
try:
    with open(CACHE_FILE_PATH, 'rb') as file:
        cache = pickle.load(file)
except FileNotFoundError:
    # Nếu tệp lưu trữ không tồn tại, khởi tạo cache trống
    cache = {}

def save_to_cache(prompt, response):
    # Lưu kết quả vào cache
    cache[prompt] = response

    # Lưu cache vào tệp
    with open(CACHE_FILE_PATH, 'wb') as file:
        pickle.dump(cache, file)

def load_from_cache(prompt):
    # Kiểm tra xem prompt có trong cache không
    if prompt in cache:
        return cache[prompt]
    else:
        return None


# GPT 3 prompt generation function
def generate_gpt_response(prompt):
    global last_request_time
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < WAIT_TIME_BETWEEN_REQUESTS:
        time.sleep(WAIT_TIME_BETWEEN_REQUESTS - time_since_last_request)

    # Kiểm tra cache trước
    cached_response = load_from_cache(prompt)
    if cached_response:
        return cached_response

    # Thêm tham số frequency_capping để đặt giới hạn RPM và TPM
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=1200,
        n=1,
        stop=None
    )

    generated_response = response.choices[0].text.strip()

    # Lưu kết quả vào cache
    save_to_cache(prompt, generated_response)

    last_request_time = time.time()

    return generated_response

def get_model_response(msg):
    prompt = msg.lower()  # Chuyển đổi câu tin nhắn thành chữ thường để sử dụng làm key trong cache
    cached_response = load_from_cache(prompt)
    if cached_response:
        return cached_response
    else:
        model_response = getResponse(predict_class(msg, model), intents)
        save_to_cache(prompt, model_response)
        return model_response

def chatbot_response(msg):
    # Tính toán câu trả lời của model và của GPT-3
    model_response = get_model_response(msg)
    gpt_response = generate_gpt_response(msg)
    
    # Nếu không có câu trả lời từ model, sử dụng GPT-3
    if not model_response:
        return gpt_response
    
    return gpt_response
    
