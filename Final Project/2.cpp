#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>

using namespace std;

void scheduleJobs(int n, const vector<pair<int, int>>& dependencies) {
    // Create a hash map to store dependencies for each job
    unordered_map<int, vector<int>> dependents;
    // Create a hash map to store the count of dependencies for each job
    unordered_map<int, int> dependencyCount;

    // Populate dependents and dependencyCount hash maps
    for (const auto& dependency : dependencies) {
        int job = dependency.first;
        int dependent = dependency.second;
        dependents[dependent].push_back(job);
        dependencyCount[job]++;
    }

    // Initialize the ready queue with jobs that have no dependencies
    queue<int> readyQueue;
    for (int job = 0; job < n; ++job) {
        if (dependencyCount.find(job) == dependencyCount.end()) {
            readyQueue.push(job);
        }
    }

    // Process the jobs
    while (!readyQueue.empty()) {
        // Select a job from the ready queue
        int currentJob = readyQueue.front();
        readyQueue.pop();
        cout << "Executing Job " << currentJob << endl;

        // Update dependency count for jobs that depend on the current job
        for (int dependentJob : dependents[currentJob]) {
            dependencyCount[dependentJob]--;

            // If the dependency count becomes zero, add the job to the ready queue
            if (dependencyCount[dependentJob] == 0) {
                readyQueue.push(dependentJob);
            }
        }
    }
}

int main() {

    int n = 5;
    vector<pair<int, int>> dependencies = {{1, 4}, {2, 3}, {3, 1}, {4, 0}};

    scheduleJobs(n, dependencies);

    return 0;
}
