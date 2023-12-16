#include <iostream>
using namespace std;

// Node structure for the circular linked list
struct Node {
    int data;
    Node* next;
    
    Node(int val) : data(val), next(nullptr) {}
};

// Function to find the last remaining participant
int findLastRemaining(int n, int m) {
    
    // Create a circular linked list
    Node* head = new Node(0);
    Node* current = head;
    for (int i = 1; i < n; ++i) {
        current->next = new Node(i);
        current = current->next;
    }
    
    // Connect the last node to the head, making it circular
    current->next = head;
    
    // Game running
    Node* prev = nullptr;
    current = head;
    int count = 0;
    while (current->next != current) {
        count++;
        if (count == m) {
            // Remove the m-th person
            if (prev == nullptr) {
                head = current->next;
            } else {
                prev->next = current->next;
            }
            Node* temp = current;
            current = current->next;
            delete temp;
            count = 0;
        } else {
            prev = current;
            current = current->next;
        }
    }
    
    return current->data;
}

int main() {
    int n, m;
    cout << "Enter the number of participants (n): ";
    cin >> n;
    cout << "Enter the counting interval larger than 1 (m): ";
    cin >> m;
    
    int winnerPosition = findLastRemaining(n, m);
    
    cout << "The winner is at position: " << winnerPosition << endl;
    
    return 0;
}
