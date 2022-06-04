#1. Set up
'''
setup

    Create a Watson Studio Service.

    Create a Watson Machine Learning (WML) Service instance (a free plan is offered and information about how to create the instance is here).

    Create a Cloud Object Storage (COS) instance (a lite plan is offered and information about how to order storage is here).
    Note: When using Watson Studio, you already have a COS instance associated with the project you are running the notebook in.

    Create new credentials with HMAC:

        Go to Watson Studios, click the button on the top left corner.

        Select Data Services

        Go to your COS(Cloud Object Storage) dashboard.

        In the Service credentials tab, click New Credential+.

        Add the inline configuration parameter: {"HMAC":true}, click Add. (For more information, see HMAC.)

        This configuration parameter adds the following section to the instance credentials, (for use later in this notebook):

        "cos_hmac_keys": {
              "access_key_id": "-------",
              "secret_access_key": "-------"
         }

'''