def delete_path_from_nexus(nexusObject, path):
    try:
        del nexusObject[path]
    except:
        pass