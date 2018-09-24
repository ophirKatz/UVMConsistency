/************************************************************
 *                                                          *
 *  This program's goal is to proof that CUDA's             *
 *  UVM consistency model is strong only when local         *
 *  thread memory is flushed before handling page-          *
 *  fault in CPU. And it shows that when this condition     *
 *  is not adhered to, the consistency lacks strength.      *
 *                                                          *
 *  The project showcases a real-life example of usage      *
 *  of the UVM model, and presents a critical bug           *
 *  caused by the consistency model issues.                 *
 *                                                          *
 *                                                          *
 *  The example at hand - a bank server, that receives      *
 *  clients' requests that deposit an amount of money into  *
 *  their account. The bug occurs when the balance is       *
 *  later checked and the client discovers it has not       *
 *  changed. The fix is to add the nessesary actions in     *
 *  the code to insure the balance is correct.              *
 *                                                          *
 ************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>

using namespace std;

#define CUDA_CHECK(f) do {                                                                \
  cudaError_t e = f;                                                                      \
  if (e != cudaSuccess) {                                                                 \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(1);                                                                            \
  }                                                                                       \
} while (0)

#define ONLY_THREAD if (threadIdx.x == 0)

#define UVMSPACE      volatile
#define START         0
#define ALERT_CPU     1
#define ALERT_GPU     2

#define OUT

#define NUM_BANK_ACCOUNTS 1

typedef unsigned long long int ulli;

class ManagedBankAccount {
private:
  static void *allocate(size_t size) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    CUDA_CHECK(cudaDeviceSynchronize());
    return ptr;
  }
public:

  static void initialize_account(UVMSPACE ManagedBankAccount *account,
                                unsigned long balance, unsigned long id) {
    account->balance = balance;
    account->account_id = id;
    __sync_synchronize();
  }

  ManagedBankAccount() : balance(0), account_id(0) {}
  
  void *operator new(size_t size) {
    return allocate(size);
  }

  void *operator new[](size_t size) {
    return allocate(size);
  }

  void operator delete[](void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }

  UVMSPACE unsigned long balance;
  UVMSPACE unsigned long account_id;
};



/**
    The kernel that performs the deposit
*/

__device__ void deposit_to_account(UVMSPACE ManagedBankAccount *bank_account, unsigned long deposit_amount,
                                  volatile int *finished) {
  // NOTE : need to sync this for the change to be seen in CPU (this is whats being tested)
  atomicAdd((ulli *) &bank_account->balance, (ulli) deposit_amount);
  *finished = ALERT_CPU;
  // __threadfence_system(); // Writing finished GPU memory so CPU can see

  // Wait for CPU to release
  while (*finished != ALERT_GPU);
  
  printf(" --- --- --- Finished kernel --- --- --- \n");
}

__global__ void bank_deposit(UVMSPACE void *bank_ptr, unsigned long account_id, unsigned long deposit_amount,
                            volatile int *finished, UVMSPACE OUT int *status) {
  UVMSPACE ManagedBankAccount *account = (ManagedBankAccount *) bank_ptr;
  int account_index = 0;
  for (; account_index < NUM_BANK_ACCOUNTS; account_index++, account++) {
    UVMSPACE unsigned long id = account->account_id;
    if (id == account_id) {
      break;
    }
  }
  if (account_index >= NUM_BANK_ACCOUNTS) { // Account was not found (error handling...)
    *status = -1;
    __threadfence_system();
    return;
  }
  *status = 0;
  __threadfence_system();

  deposit_to_account(account, deposit_amount, finished);
}


class ManagedBank {
public:

  ManagedBank() {
    accounts = new ManagedBankAccount[NUM_BANK_ACCOUNTS];
    for (int i = 0; i < NUM_BANK_ACCOUNTS; i++) {
      unsigned long balance = (i + 1) * 1000;
      unsigned long id = i + 1;
      ManagedBankAccount::initialize_account(accounts + i, balance, id);
    }

    CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int)));
    memset((void *) finished, START, sizeof(int));

    // Writing all the changes of UM to GPU
    __sync_synchronize();
  }

  ~ManagedBank() {
    cout << endl << "Destroying Bank" << endl;
    cout << "Freeing <accounts> array" << endl;
    delete[] accounts;
    cout << "Freeing <finished>" << endl;
    CUDA_CHECK(cudaFree((int *) finished));
  }

  void start_transaction() {
    *finished = START;
    __sync_synchronize();
  }

  bool deposit(unsigned long account_id, unsigned long deposit_amount) {
    UVMSPACE int *action_status;
    CUDA_CHECK(cudaMallocManaged(&action_status, sizeof(int)));

    // Call the kernel that uses the unified memory mapped page
    bank_deposit<<<1,1>>>(accounts, account_id, deposit_amount, finished, action_status);

    return true;
  }

  long check_balance(unsigned long account_id) {
    long balance = -1;
    UVMSPACE ManagedBankAccount *account = (ManagedBankAccount *) accounts;

    for (int account_index = 0; account_index < NUM_BANK_ACCOUNTS; account_index++, account++) {
      if (account->account_id == account_id) {
        balance = account->balance;
        break;
      }
    }

    return balance;
  }

  void wait_for_deposit() {
    while(*finished != ALERT_CPU);
  }

  void finish_deposit() {
    cout << "[in finish_deposit] finished was = " << *finished << " and now = " << ALERT_GPU << endl;
    *finished = ALERT_GPU; // check_balance means the CPU has its result
    __sync_synchronize();
    CUDA_CHECK(cudaDeviceSynchronize());  // Waiting for kernel to finish
  }

  void print() {
    cout << "******  UVM Managed Bank  ******" << endl
         << "\tAccounts : " << endl;
    cout << "\t" << "Account id   |   Account balance" << endl;
    UVMSPACE ManagedBankAccount *account = (ManagedBankAccount *) accounts;
    for (int i = 0; i < NUM_BANK_ACCOUNTS; i++, account++) {
      cout << "\t" << "      " << account->account_id << "      |      " << account->balance << endl;
    }
    cout << endl;
  }

private:
  UVMSPACE ManagedBankAccount *accounts;
  volatile int *finished;
};


/**
    The class that represents the consistency model
*/

class UVMConsistency {
public:
  static void strong_consistency_broken() {
    ManagedBank bank;
    bank.print();

    // Getting initial balance
    unsigned long balance = bank.check_balance(UVMConsistency::account_id);

    // Starting the transaction
    bank.start_transaction();
    // Depositing an amount - done in GPU thread
    bank.deposit(UVMConsistency::account_id, UVMConsistency::deposit_amount);
    
    unsigned long new_balance = balance + UVMConsistency::deposit_amount;
    // Wait for deposit to occur in GPU thread
    bank.wait_for_deposit();
    bank.print();
    unsigned long second_balance = bank.check_balance(UVMConsistency::account_id);
    
    // Finish the transaction
    bank.finish_deposit();
    
    // Check the result
    if (second_balance != new_balance) {
      cout << "Error! Strong Consistency is not adhered to!" << endl;
    } else {
      cout << "Success! Strong Consistency achieved!" << endl;
    }
  }
private:
  static const unsigned long account_id = 1;
  static const unsigned long deposit_amount = 1000;
};

int main() {
  UVMConsistency::strong_consistency_broken();
  return 0;
}