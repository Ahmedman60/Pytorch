
    def minOperations(self, logs: List[str]) -> int:
        files = []
        for i in logs:
            if i != "../":
                files.append(i)
            elif i == "./":
                pass
            else:
                # it is a file
                if files:
                    files.pop()

        return len(files)