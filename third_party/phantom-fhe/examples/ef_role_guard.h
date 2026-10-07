#pragma once

#ifdef EF_PIPE_CKKS_CLIENT_INCLUDED
#error "server-role binary includes pipe_ckks_client.h (secret-key helpers)"
#endif

#define EF_SERVER_ROLE 1
#define PhantomSecretKey EF_SERVER_ROLE_FORBIDS_PhantomSecretKey
#define encrypt_packed_pairs EF_SERVER_ROLE_FORBIDS_encrypt_packed_pairs
#define decrypt_blocks EF_SERVER_ROLE_FORBIDS_decrypt_blocks
