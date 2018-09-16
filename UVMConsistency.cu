/************************************************************
 *                                                          *
 *  This program's goal is to prove that CUDA's             *
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
#define THREAD_FINISH 1
#define CPU_FINISH    2
#define OUT

#define NUM_BANK_ACCOUNTS 1


class ManagerBankAccount {
public:

  static void initialize_account(UVMSPACE ManagerBankAccount *account,
                                unsigned long balance, unsigned long id) {
    // TODO : Add flushes? syncs?
    account->balance = balance;
    account->account_id = id;
  }

  ManagerBankAccount() : balance(0), account_id(0) {}

  void *operator new() {
    void *ptr;
    size_t len = sizeof(ManagerBankAccounts);
    CUDA_CHECK(cudaMallocManaged(&ptr, len));
    CUDA_CHECK(cudaDeviceSynchronize());
    return ptr;
  }

  void operator delete(void *ptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(ptr));
  }

private:
  unsigned long balance;
  unsigned long account_id;
};



/**
    The kernel that performs the deposit
*/

__device__ void deposit_to_account(UVMSPACE ManagerBankAccount *bank_account, unsigned long deposit_amount,
                                  volatile int *finished) {
  // FIXME : Needs to be wrapped with flushes
  ONLY_THREAD {
    bank_account->balance += deposit_amount;
    *finished = THREAD_FINISH;
  }
  // Wait for CPU to release
  while (*finished != CPU_FINISH);
}

__global__ void bank_deposit(UVMSPACE void *bank_ptr, unsigned long account_id, unsigned long deposit_amount,
                            volatile int *finished, UVMSPACE OUT int *status) {
  UVMSPACE ManagerBankAccount *account = (ManagerBankAccount *) bank_ptr;
  for (int account_index = 0; account_index < NUM_BANK_ACCOUNTS; account_index++, account++) {
    if (account->account_id == account_id) {
      break;
    }
  }
  if (account_index >= NUM_BANK_ACCOUNTS) { // Account was not found (error handling...)
    *status = -1;
    return;
  }
  *status = 0;
  deposit_to_account(account, deposit_amount, finished);
}


class ManagedBank {
public:

  ManagedBank() {
    accounts = new ManagerBankAccount[NUM_BANK_ACCOUNTS];
    for (int i = 0; i < NUM_BANK_ACCOUNTS; i++) {
      ManagerBankAccount::initialize_account(accounts + i, i * 1000, i);
    }

    CUDA_CHECK(cudaMallocManaged(&finished, sizeof(int)));
    memset(finished, 0, sizeof(int));

    __sync_synchronize();
  }

  bool deposit(unsigned long account_id, unsigned long deposit_amount) {
    UVMSPACE int *action_status;
    CUDA_CHECK(cudaMallocManaged(&action_status, sideof(int)));

    // Call the kernel that uses the unified memory mapped page
    bank_deposit<<<1,1>>>(accounts, account_id, deposit_amount, finished, action_status);
  }

   long check_balance(unsigned long account_id) {
    UVMSPACE ManagerBankAccount *account = (ManagerBankAccount *) bank_ptr;
    for (int account_index = 0; account_index < NUM_BANK_ACCOUNTS; account_index++, account++) {
      if (account->account_id == account_id) {
        return account->balance;
      }
    }
    return -1;
  }

private:
  UVMSPACE ManagerBankAccount *accounts;
  volatile int *finished;
};


/**
    The class that represents the consistency model
*/

class UVMConsistency {
  static void strong_consistency() {

  }

  static void strong_consistency_broken() {
    
  }
};
