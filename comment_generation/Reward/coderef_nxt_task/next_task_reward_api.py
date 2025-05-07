from flask import Flask, request, jsonify
from next_task_reward import SubsequentTaskReward, NEXT_TASK_REWARD_TYPES

app = Flask(__name__)
reward_model = None

@app.route('/compute', methods=['POST'])
def get_compute():
    if request.is_json:
        batch = request.get_json()
        result = reward_model.compute_reward(batch)
        return jsonify({"reward": result})
    else:
        print('>ERROR OCCURED!')
        return jsonify({"error": "Request must be JSON"}), 400
    
if __name__ == '__main__':
    print(reward_model, 'Server started')
    if not reward_model:
        reward_model = SubsequentTaskReward()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
